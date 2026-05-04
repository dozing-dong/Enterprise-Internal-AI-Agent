from backend.rag.rewrite import rewrite_question_for_retrieval


def test_rewrite_adds_travel_policy_hints_for_chinese_query():
    # Chinese phrasing kept as Unicode escapes; this test exercises the
    # bilingual hint expansion in the rewrite module.
    question = (
        "\u6211\u53eb Alice Carter\uff0c\u8981\u53bb\u5916\u5730\u51fa\u5dee\uff0c"
        "\u6839\u636e\u6211\u7684\u516c\u53f8\u4fe1\u606f\u751f\u6210\u5e94\u8be5\u6ce8\u610f\u7684 policy"
    )
    retrieval_question = rewrite_question_for_retrieval(question, rewrite_chain=None)
    assert "business travel policy" in retrieval_question
    assert "travel request" in retrieval_question
    assert "company policy" in retrieval_question


def test_rewrite_adds_expense_hints_when_reimbursement_mentioned():
    # Chinese phrasing kept as Unicode escapes for the same reason.
    question = "\u51fa\u5dee\u62a5\u9500\u6d41\u7a0b\u662f\u4ec0\u4e48\uff1f"
    retrieval_question = rewrite_question_for_retrieval(question, rewrite_chain=None)
    assert "expense reimbursement policy" in retrieval_question
    assert "approval workflow" in retrieval_question
