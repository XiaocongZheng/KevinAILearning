import unittest
from types import SimpleNamespace

from study_agents import (
    StudyAssistant,
    StudyProgress,
    classify_intent,
    parse_answer_choice,
)


class FakeRetriever:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else [
            SimpleNamespace(
                page_content="Strong winds can reduce control of the aircraft.",
                metadata={"source": "manual.pdf", "page": 2},
            )
        ]
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


class FakeLlm:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=next(self.responses))


class StudyAgentTests(unittest.TestCase):
    def test_routes_study_plan_and_quiz_requests_to_specialists(self):
        self.assertEqual(
            classify_intent("我完全不懂，第一天该学什么"), "roadmap"
        )
        self.assertEqual(
            classify_intent("出一道关于大风天气飞行的模拟题"), "quiz"
        )
        self.assertEqual(classify_intent("继续"), "quiz")
        self.assertEqual(classify_intent("什么是 VLOS？"), "explain")

    def test_answer_choice_accepts_short_and_natural_chinese_answers(self):
        self.assertEqual(parse_answer_choice("B"), "B")
        self.assertEqual(parse_answer_choice("我选答案是 C"), "C")
        self.assertIsNone(parse_answer_choice("我不确定"))

    def test_wrong_quiz_answer_triggers_rag_remediation_and_tracks_weak_topic(self):
        llm = FakeLlm(
            [
                "QUESTION: 大风时应如何操作？\n"
                "A) 继续按原计划飞行\n"
                "B) 评估风险并避免不安全飞行\n"
                "C) 忽略风速\n"
                "D) 提高飞行高度\n"
                "CORRECT_ANSWER: B\n"
                "RATIONALE: 强风会影响飞行器控制。",
                "强风会降低飞行器的可控性，应先评估风险。",
            ]
        )
        retriever = FakeRetriever()
        assistant = StudyAssistant(llm, retriever)
        progress = StudyProgress()

        question = assistant.respond(
            "出一道关于大风天气飞行的模拟题", progress
        )
        self.assertTrue(question.waiting_for_answer)
        self.assertNotIn("CORRECT_ANSWER", question.text)
        self.assertEqual(retriever.queries[0], "大风天气飞行")

        response = assistant.respond("A", progress)

        self.assertEqual(
            response.agent_name, "考官 + 知识拆解与翻译教员"
        )
        self.assertIn("补课讲解", response.text)
        self.assertEqual(progress.attempts, 1)
        self.assertEqual(progress.correct_answers, 0)
        self.assertEqual(progress.weak_topics, ["大风天气飞行"])
        self.assertEqual(retriever.queries[1], "大风天气飞行")

    def test_invalid_quiz_answer_keeps_question_pending(self):
        llm = FakeLlm(
            [
                "QUESTION: 题目？\nA) 甲\nB) 乙\nC) 丙\nD) 丁\n"
                "CORRECT_ANSWER: A\nRATIONALE: 解释。",
            ]
        )
        assistant = StudyAssistant(llm, FakeRetriever())
        progress = StudyProgress()
        assistant.respond("出一道关于空域的模拟题", progress)

        response = assistant.respond("不清楚", progress)

        self.assertTrue(response.waiting_for_answer)
        self.assertIsNotNone(progress.pending_quiz)
        self.assertEqual(progress.attempts, 0)

    def test_correct_answer_clears_a_previously_weak_topic(self):
        llm = FakeLlm(
            [
                "QUESTION: 题目？\nA) 甲\nB) 乙\nC) 丙\nD) 丁\n"
                "CORRECT_ANSWER: A\nRATIONALE: 解释。",
            ]
        )
        assistant = StudyAssistant(llm, FakeRetriever())
        progress = StudyProgress(weak_topics=["空域与法规"])
        assistant.respond("出一道关于空域与法规的模拟题", progress)

        response = assistant.respond("我选 A", progress)

        self.assertIn("回答正确", response.text)
        self.assertEqual(progress.correct_answers, 1)
        self.assertEqual(progress.weak_topics, [])

    def test_failed_remediation_keeps_the_quiz_pending_for_retry(self):
        llm = FakeLlm(
            [
                "QUESTION: 题目？\nA) 甲\nB) 乙\nC) 丙\nD) 丁\n"
                "CORRECT_ANSWER: B\nRATIONALE: 解释。",
                None,
            ]
        )
        assistant = StudyAssistant(llm, FakeRetriever())
        progress = StudyProgress()
        assistant.respond("出一道关于空域的模拟题", progress)

        with self.assertRaises(TypeError):
            assistant.respond("A", progress)

        self.assertIsNotNone(progress.pending_quiz)
        self.assertEqual(progress.attempts, 0)

    def test_quiz_without_retrieved_material_does_not_invent_a_question(self):
        assistant = StudyAssistant(FakeLlm([]), FakeRetriever(documents=[]))

        response = assistant.respond(
            "出一道关于空域的模拟题", StudyProgress()
        )

        self.assertFalse(response.waiting_for_answer)
        self.assertIn("先不生成", response.text)


if __name__ == "__main__":
    unittest.main()
