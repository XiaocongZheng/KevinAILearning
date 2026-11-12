from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    from langchain_core.documents import Document


@dataclass
class QuizQuestion:
    topic: str
    correct_answer: str
    rationale: str


@dataclass
class StudyProgress:
    attempts: int = 0
    correct_answers: int = 0
    weak_topics: List[str] = field(default_factory=list)
    pending_quiz: Optional[QuizQuestion] = None

    def record_answer(self, topic: str, is_correct: bool) -> None:
        self.attempts += 1
        if is_correct:
            self.correct_answers += 1
            self.weak_topics = [
                weak_topic
                for weak_topic in self.weak_topics
                if weak_topic.casefold() != topic.casefold()
            ]
        elif not any(
            weak_topic.casefold() == topic.casefold()
            for weak_topic in self.weak_topics
        ):
            self.weak_topics.append(topic)


@dataclass
class AgentResponse:
    agent_name: str
    text: str
    sources: List[str] = field(default_factory=list)
    waiting_for_answer: bool = False


def classify_intent(message: str) -> str:
    if re.search(
        r"模拟题|出一道|测验|考题|练习题|选择题|考官|测试我|继续|下一题|再来一题|quiz|question|continue|next",
        message,
        re.IGNORECASE,
    ):
        return "quiz"
    if re.search(
        r"学习计划|学习大纲|大纲|第一天|今天学什么|零基础|备考计划|规划|outline|study plan",
        message,
        re.IGNORECASE,
    ):
        return "roadmap"
    return "explain"


def parse_answer_choice(message: str) -> Optional[str]:
    normalized = message.strip().upper()
    if normalized in {"A", "B", "C", "D"}:
        return normalized

    match = re.search(
        r"(?:答案|选择|选|ANSWER|CHOICE)\s*[:：是为]?\s*([ABCD])\b",
        normalized,
    )
    if match is None:
        match = re.match(r"\s*([ABCD])(?:\b|[.、，])", normalized)
    return match.group(1) if match is not None else None


def _format_context(documents: Sequence[Document]) -> str:
    if not documents:
        return "在现有资料库中未检索到相关内容。"

    passages = []
    for document in documents:
        source = Path(str(document.metadata.get("source", "未知文件"))).name
        page = document.metadata.get("page")
        page_reference = (
            f"第 {page + 1} 页" if isinstance(page, int) else "页码未知"
        )
        passages.append(
            f"[资料来源：{source}，{page_reference}]\n{document.page_content}"
        )
    return "\n\n".join(passages)


def _source_references(documents: Sequence[Document]) -> List[str]:
    references = []
    for document in documents:
        source = Path(str(document.metadata.get("source", "未知文件"))).name
        page = document.metadata.get("page")
        page_reference = (
            f"第 {page + 1} 页" if isinstance(page, int) else "页码未知"
        )
        reference = f"《{source}》{page_reference}"
        if reference not in references:
            references.append(reference)
    return references


def _message_content(result: object) -> str:
    content = getattr(result, "content", None)
    if not isinstance(content, str):
        raise TypeError("模型响应 content 必须是文本。")
    return content.strip()


class StudyAssistant:
    """Routes each turn to one of three specialist agents sharing the RAG corpus."""

    def __init__(self, llm: object, retriever: object) -> None:
        self.llm = llm
        self.retriever = retriever

    def respond(self, message: str, progress: StudyProgress) -> AgentResponse:
        if progress.pending_quiz is not None:
            return self._grade_answer(message, progress)

        intent = classify_intent(message)
        if intent == "quiz":
            return self._start_quiz(message, progress)
        if intent == "roadmap":
            return self._create_roadmap(message, progress)
        return self._explain(message)

    def _retrieve(self, query: str) -> Sequence[Document]:
        documents = self.retriever.invoke(query)
        return documents

    def _create_roadmap(
        self, message: str, progress: StudyProgress
    ) -> AgentResponse:
        documents = self._retrieve(
            "加拿大无人机 Basic 和 Advanced 考试要求、空域、法规、气象、飞行原理、"
            "操作程序及应急处置学习资料"
        )
        weak_topics = "、".join(progress.weak_topics) or "尚无错题记录"
        prompt = (
            "你是“导航规划教员”，负责带领零基础学员备考加拿大无人机飞行考试。"
            "先用简明中文说明 Basic 与 Advanced 考试的核心区别、VLOS 的含义；"
            "如果检索资料不能支持某项具体考试要求，请明确标为待核实，不要臆测。"
            "结合资料库内容制定一个 2-4 周、按模块安排的学习计划，并明确今天的"
            "第一项任务和完成标准。结合学员薄弱主题安排复习。结尾引导学员可继续"
            "询问术语，或要求一道模拟题。不要声称已验证具体地点的空域。"
            f"\n\n学员本次需求：{message}"
            f"\n\n学员薄弱主题：{weak_topics}"
            f"\n\n检索资料：\n{_format_context(documents)}"
        )
        return AgentResponse(
            "导航规划教员", _message_content(self.llm.invoke(prompt)),
            _source_references(documents),
        )

    def _explain(self, message: str) -> AgentResponse:
        documents = self._retrieve(message)
        prompt = (
            "你是“知识拆解与翻译教员”。优先且仅依据检索资料回答用户的问题。"
            "用容易理解的中文解释概念，列出关键英文术语及准确中文对照；必要时"
            "引用短句并解释，不要伪造原文。指出资料文件和页码。资料不足时，明确"
            "说明“在现有的无人机学习资料中未找到该内容”，并区分一般背景知识与"
            "资料证据。涉及空域或法规时，不要根据地名猜测许可要求。"
            f"\n\n用户问题：{message}"
            f"\n\n检索资料：\n{_format_context(documents)}"
        )
        return AgentResponse(
            "知识拆解与翻译教员",
            _message_content(self.llm.invoke(prompt)),
            _source_references(documents),
        )

    def _start_quiz(
        self, message: str, progress: StudyProgress
    ) -> AgentResponse:
        topic = message.strip()
        if topic.casefold() in {"继续", "下一题", "再来一题", "continue", "next"}:
            topic = progress.weak_topics[0] if progress.weak_topics else "空域与法规"
        else:
            topic = re.sub(
                r"^(?:请)?(?:给我)?(?:出一道|出一题|来一道)(?:关于|有关)?",
                "",
                topic,
                flags=re.IGNORECASE,
            )
            topic = re.sub(
                r"(?:的)?(?:模拟题|选择题|练习题|考题|测验)$",
                "",
                topic,
                flags=re.IGNORECASE,
            ).strip(" ，,:：的")
            topic = topic or "空域与法规"
        documents = self._retrieve(topic)
        if not documents:
            return AgentResponse(
                "实战场景与考官",
                "资料库中暂时没有检索到这个主题的依据，因此我先不生成可能不准确的题目。"
                "请换一个更具体的主题，或先确认 PDF 已完成索引。",
            )
        prompt = (
            "你是“实战场景与考官”。根据检索资料生成一道加拿大无人机考试风格的"
            "单选场景题，必须有 A、B、C、D 四个选项，只生成一题。题目和选项应"
            "可由资料支持。若使用渥太华、卡尔顿大学等真实地点，不得臆断该地的"
            "实际空域或许可要求；资料没有地理证据时，把场景明确写成假设情景。"
            "严格按以下格式输出，不要输出代码围栏：\n"
            "QUESTION: 题干\nA) 选项\nB) 选项\nC) 选项\nD) 选项\n"
            "CORRECT_ANSWER: A/B/C/D\nRATIONALE: 简短解释\n"
            f"\n本题主题：{topic}"
            f"\n\n检索资料：\n{_format_context(documents)}"
        )
        generated = _message_content(self.llm.invoke(prompt))
        answer_match = re.search(
            r"(?im)^CORRECT_ANSWER:\s*([ABCD])\s*$", generated
        )
        rationale_match = re.search(
            r"(?im)^RATIONALE:\s*(.+)$", generated
        )
        options = re.findall(r"(?im)^[A-D][).]\s*.+$", generated)
        if (
            answer_match is None
            or rationale_match is None
            or len(options) != 4
        ):
            raise ValueError(
                "考官生成的题目格式无效；请重试生成题目。"
            )

        question_text = generated[: answer_match.start()].strip()
        progress.pending_quiz = QuizQuestion(
            topic=topic,
            correct_answer=answer_match.group(1),
            rationale=rationale_match.group(1).strip(),
        )
        return AgentResponse(
            "实战场景与考官",
            f"{question_text}\n\n请回复 A、B、C 或 D 作答。",
            _source_references(documents),
            waiting_for_answer=True,
        )

    def _grade_answer(
        self, message: str, progress: StudyProgress
    ) -> AgentResponse:
        quiz = progress.pending_quiz
        if quiz is None:
            raise RuntimeError("没有待批改的模拟题。")
        choice = parse_answer_choice(message)
        if choice is None:
            return AgentResponse(
                "实战场景与考官",
                "我还没有识别到选项，请直接回复 A、B、C 或 D。",
                waiting_for_answer=True,
            )

        is_correct = choice == quiz.correct_answer
        result = "回答正确！" if is_correct else (
            f"这题选错了。正确答案是 {quiz.correct_answer}。"
        )
        if is_correct:
            progress.pending_quiz = None
            progress.record_answer(quiz.topic, True)
            return AgentResponse(
                "实战场景与考官",
                f"{result}\n\n考官解析：{quiz.rationale}\n"
                f"当前答对 {progress.correct_answers}/{progress.attempts} 题。",
            )

        documents = self._retrieve(quiz.topic)
        prompt = (
            "你是知识拆解与翻译教员，正在为一道答错的模拟题进行补课。"
            "用浅显中文解释正确答案和关键概念，点出常见误区，并严格根据给定"
            "资料说明依据；资料不支持的内容要明确指出。"
            f"\n\n题目主题：{quiz.topic}"
            f"\n正确答案：{quiz.correct_answer}"
            f"\n考官解析：{quiz.rationale}"
            f"\n学员选择：{choice}"
            f"\n\n检索资料：\n{_format_context(documents)}"
        )
        explanation = _message_content(self.llm.invoke(prompt))
        progress.pending_quiz = None
        progress.record_answer(quiz.topic, False)
        return AgentResponse(
            "考官 + 知识拆解与翻译教员",
            f"{result}\n\n补课讲解：\n{explanation}\n\n"
            f"该知识点已加入薄弱主题；答对 {progress.correct_answers}/"
            f"{progress.attempts} 题。你可以让考官围绕薄弱主题再出一题。",
            _source_references(documents),
        )
