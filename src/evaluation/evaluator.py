"""
RAG评估器：LLM-as-a-Judge 评估系统

功能:
- 使用高阶模型评估RAG系统的回答质量
- 支持 Faithfulness (忠实度) 和 Relevancy (相关性) 评估
- 批量评估和报告生成
"""
import os
import re
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# 评估 Prompt 模板
EVALUATION_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

CONTEXT (Retrieved Documents):
{context}

USER QUESTION:
{question}

MODEL ANSWER:
{answer}

TASK: Evaluate the answer on these dimensions:

1. FAITHFULNESS (0-10): Does the answer strictly follow the context? Are there any hallucinations (facts not supported by context)?
   - 10: Perfectly faithful, all claims supported by context
   - 7-9: Mostly faithful, minor unsupported details
   - 4-6: Partially faithful, some hallucinations
   - 0-3: Significant hallucinations or contradictions

2. RELEVANCY (0-10): Does the answer directly address the user's question?
   - 10: Directly and completely answers the question
   - 7-9: Answers the question with minor tangents
   - 4-6: Partially addresses the question
   - 0-3: Does not answer the question

3. CITATION_QUALITY (0-10): Does the answer cite sources naturally?
   - 10: Natural, professional citations (e.g., "The Q3 report shows...")
   - 7-9: Some citations, mostly natural
   - 4-6: Citations present but awkward or mechanical
   - 0-3: No citations or very poor citation style

OUTPUT FORMAT: Return ONLY a single-line JSON object with integer scores 0-10. No markdown, no extra text.
Example: {{"faithfulness": 8, "faithfulness_reasoning": "Mostly accurate", "relevancy": 9, "relevancy_reasoning": "Addresses question", "citation_quality": 7, "citation_reasoning": "Some citations"}}"""

# 无上下文时的评估 Prompt
NO_CONTEXT_EVALUATION_PROMPT = """You are an expert evaluator for AI assistant responses.

USER QUESTION:
{question}

MODEL ANSWER:
{answer}

Note: No specific documents were provided for this question. The AI used its general knowledge.

TASK: Evaluate the answer on these dimensions:

1. HELPFULNESS (0-10): Is the answer helpful and informative?
   - 10: Excellent, comprehensive answer
   - 7-9: Good answer, covers main points
   - 4-6: Partially helpful
   - 0-3: Not helpful

2. RELEVANCY (0-10): Does the answer directly address the user's question?
   - 10: Directly and completely answers the question
   - 7-9: Answers the question with minor tangents
   - 4-6: Partially addresses the question
   - 0-3: Does not answer the question

3. ACCURACY (0-10): Based on your knowledge, is the answer factually correct?
   - 10: Completely accurate
   - 7-9: Mostly accurate
   - 4-6: Some inaccuracies
   - 0-3: Significant errors

OUTPUT FORMAT: Return ONLY a single-line JSON object with integer scores 0-10. No markdown, no extra text.
Example: {{"helpfulness": 8, "helpfulness_reasoning": "Good explanation", "relevancy": 9, "relevancy_reasoning": "Addresses question", "accuracy": 7, "accuracy_reasoning": "Mostly correct"}}"""


@dataclass
class EvaluationResult:
    """单个评估结果"""
    question: str
    answer: str
    context: str
    has_context: bool
    # 有上下文时的评分
    faithfulness: Optional[float] = None
    faithfulness_reasoning: Optional[str] = None
    relevancy: Optional[float] = None
    relevancy_reasoning: Optional[str] = None
    citation_quality: Optional[float] = None
    citation_reasoning: Optional[str] = None
    # 无上下文时的评分
    helpfulness: Optional[float] = None
    helpfulness_reasoning: Optional[str] = None
    accuracy: Optional[float] = None
    accuracy_reasoning: Optional[str] = None
    # 元信息
    error: Optional[str] = None

    def get_avg_score(self) -> float:
        """计算平均分"""
        if self.has_context:
            scores = [s for s in [self.faithfulness, self.relevancy, self.citation_quality] if s is not None]
        else:
            scores = [s for s in [self.helpfulness, self.relevancy, self.accuracy] if s is not None]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class EvaluationReport:
    """评估报告"""
    total_cases: int
    evaluated_cases: int
    failed_cases: int
    # 有上下文的统计
    avg_faithfulness: float
    avg_relevancy: float
    avg_citation_quality: float
    # 无上下文的统计
    avg_helpfulness: float
    avg_accuracy: float
    # 总体
    overall_avg: float
    # 低分用例
    low_score_cases: List[Dict]
    # 时间戳
    timestamp: str

    def to_markdown(self) -> str:
        """生成Markdown格式的报告"""
        report = f"""# RAG System Evaluation Report
Generated: {self.timestamp}

## Summary
| Metric | Value |
|--------|-------|
| Total Cases | {self.total_cases} |
| Successfully Evaluated | {self.evaluated_cases} |
| Failed | {self.failed_cases} |
| **Overall Average Score** | **{self.overall_avg:.2f}/10** |

## Context-Based Answers
| Dimension | Average Score |
|-----------|---------------|
| Faithfulness | {self.avg_faithfulness:.2f}/10 |
| Relevancy | {self.avg_relevancy:.2f}/10 |
| Citation Quality | {self.avg_citation_quality:.2f}/10 |

## General Knowledge Answers
| Dimension | Average Score |
|-----------|---------------|
| Helpfulness | {self.avg_helpfulness:.2f}/10 |
| Relevancy | {self.avg_relevancy:.2f}/10 |
| Accuracy | {self.avg_accuracy:.2f}/10 |

## Low Score Cases (< 7.0)
"""
        if self.low_score_cases:
            for case in self.low_score_cases:
                report += f"\n### Question: {case['question'][:100]}...\n"
                report += f"- **Average Score**: {case['avg_score']:.2f}\n"
                report += f"- **Issue**: {case.get('main_issue', 'Low overall score')}\n"
        else:
            report += "\n*No low score cases found!*\n"

        report += """
## Recommendations
Based on the evaluation results:
"""
        # 动态生成建议
        if self.avg_faithfulness < 7:
            report += "- Consider adjusting the similarity threshold to improve retrieval quality\n"
        if self.avg_citation_quality < 7:
            report += "- Update system prompt to encourage more natural source citations\n"
        if self.avg_relevancy < 7:
            report += "- Review and refine the prompt template to better focus answers\n"

        return report


class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(
        self,
        judge_model: str = "gemini-2.5-flash",  # Flash is more reliable for structured JSON output
        temperature: float = 0.1  # 低温度保证评估一致性
    ):
        """
        初始化评估器

        Args:
            judge_model: 用于评估的模型
            temperature: 模型温度 (低温度保证一致性)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("请设置GOOGLE_API_KEY环境变量")

        self.client = genai.Client(api_key=api_key)
        self.judge_model = judge_model
        self.temperature = temperature

    def _call_judge(self, prompt: str) -> Dict:
        """调用评估模型"""
        text = ""
        try:
            response = self.client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=1000
                )
            )

            # Handle None response
            if response.text is None:
                return {"error": "Empty response from model"}

            # 解析JSON响应
            text = response.text.strip()

            # 处理可能的markdown代码块
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]

            text = text.strip()

            # 找到JSON对象的开始和结束
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx + 1]

            # Try to parse, if fails try to fix common issues
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Try to extract scores using regex as fallback
                scores = {}
                for key in ['faithfulness', 'relevancy', 'citation_quality',
                           'helpfulness', 'accuracy']:
                    match = re.search(rf'"{key}":\s*(\d+)', text)
                    if match:
                        scores[key] = int(match.group(1))
                        scores[f'{key}_reasoning'] = "Score extracted via regex fallback"

                if scores:
                    return scores
                raise

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": text[:300] if text else "empty"}
        except Exception as e:
            return {"error": str(e)}

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str = ""
    ) -> EvaluationResult:
        """
        评估单个问答对

        Args:
            question: 用户问题
            answer: 模型回答
            context: 检索到的上下文 (可为空)

        Returns:
            评估结果
        """
        has_context = bool(context.strip())

        if has_context:
            prompt = EVALUATION_PROMPT.format(
                context=context,
                question=question,
                answer=answer
            )
        else:
            prompt = NO_CONTEXT_EVALUATION_PROMPT.format(
                question=question,
                answer=answer
            )

        scores = self._call_judge(prompt)

        if "error" in scores:
            return EvaluationResult(
                question=question,
                answer=answer,
                context=context,
                has_context=has_context,
                error=scores["error"]
            )

        if has_context:
            return EvaluationResult(
                question=question,
                answer=answer,
                context=context,
                has_context=has_context,
                faithfulness=scores.get("faithfulness"),
                faithfulness_reasoning=scores.get("faithfulness_reasoning"),
                relevancy=scores.get("relevancy"),
                relevancy_reasoning=scores.get("relevancy_reasoning"),
                citation_quality=scores.get("citation_quality"),
                citation_reasoning=scores.get("citation_reasoning")
            )
        else:
            return EvaluationResult(
                question=question,
                answer=answer,
                context=context,
                has_context=has_context,
                helpfulness=scores.get("helpfulness"),
                helpfulness_reasoning=scores.get("helpfulness_reasoning"),
                relevancy=scores.get("relevancy"),
                relevancy_reasoning=scores.get("relevancy_reasoning"),
                accuracy=scores.get("accuracy"),
                accuracy_reasoning=scores.get("accuracy_reasoning")
            )

    def evaluate_rag_system(
        self,
        qa_engine,
        test_cases: List[Dict],
        delay_between_calls: float = 0.5  # 0.5s is fine for paid tier (1000 req/min)
    ) -> tuple[List[EvaluationResult], EvaluationReport]:
        """
        评估整个RAG系统

        Args:
            qa_engine: RAGQAEngine实例
            test_cases: 测试用例列表，每个包含 'question' 字段
            delay_between_calls: API调用间隔 (避免限流)

        Returns:
            (评估结果列表, 评估报告)
        """
        results: List[EvaluationResult] = []

        for i, test_case in enumerate(test_cases):
            print(f"Evaluating case {i+1}/{len(test_cases)}: {test_case['question'][:50]}...")

            # 获取RAG系统的回答
            rag_result = qa_engine.answer_question(test_case['question'])
            answer = rag_result['answer']

            # 构建上下文
            relevant_chunks = rag_result.get('relevant_chunks', [])
            context = "\n\n".join([
                f"[Source: {chunk.get('source_file', 'Unknown')}, Page {chunk.get('page_number', '?')}]\n{chunk['text']}"
                for chunk in relevant_chunks
            ])

            # 评估
            eval_result = self.evaluate_single(
                question=test_case['question'],
                answer=answer,
                context=context
            )
            results.append(eval_result)

            # 避免API限流
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

        # 生成报告
        report = self._generate_report(results)

        return results, report

    def _generate_report(self, results: List[EvaluationResult]) -> EvaluationReport:
        """生成评估报告"""
        # 分类结果
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        with_context = [r for r in successful if r.has_context]
        without_context = [r for r in successful if not r.has_context]

        # 计算平均分
        def safe_avg(values):
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else 0.0

        avg_faithfulness = safe_avg([r.faithfulness for r in with_context])
        avg_citation = safe_avg([r.citation_quality for r in with_context])
        avg_helpfulness = safe_avg([r.helpfulness for r in without_context])
        avg_accuracy = safe_avg([r.accuracy for r in without_context])

        # 所有relevancy (两种情况都有)
        all_relevancy = [r.relevancy for r in successful if r.relevancy is not None]
        avg_relevancy = safe_avg(all_relevancy)

        # 总体平均
        all_scores = []
        for r in successful:
            all_scores.append(r.get_avg_score())
        overall_avg = safe_avg(all_scores)

        # 找出低分用例
        low_score_cases = []
        for r in successful:
            avg_score = r.get_avg_score()
            if avg_score < 7.0:
                # 找出主要问题
                main_issue = "Low overall score"
                if r.has_context:
                    if r.faithfulness and r.faithfulness < 7:
                        main_issue = "Hallucination risk"
                    elif r.citation_quality and r.citation_quality < 7:
                        main_issue = "Poor citation style"
                else:
                    if r.accuracy and r.accuracy < 7:
                        main_issue = "Accuracy concerns"

                low_score_cases.append({
                    'question': r.question,
                    'avg_score': avg_score,
                    'main_issue': main_issue
                })

        return EvaluationReport(
            total_cases=len(results),
            evaluated_cases=len(successful),
            failed_cases=len(failed),
            avg_faithfulness=avg_faithfulness,
            avg_relevancy=avg_relevancy,
            avg_citation_quality=avg_citation,
            avg_helpfulness=avg_helpfulness,
            avg_accuracy=avg_accuracy,
            overall_avg=overall_avg,
            low_score_cases=low_score_cases[:10],  # 最多显示10个
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def save_results(
        self,
        results: List[EvaluationResult],
        report: EvaluationReport,
        output_dir: str = "evaluation_results"
    ):
        """
        保存评估结果

        Args:
            results: 评估结果列表
            report: 评估报告
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果 (JSON)
        results_path = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {results_path}")

        # 保存报告 (Markdown)
        report_path = os.path.join(output_dir, f"report_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())
        print(f"Report saved to: {report_path}")

        # 保存报告摘要 (JSON)
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"Summary saved to: {summary_path}")
