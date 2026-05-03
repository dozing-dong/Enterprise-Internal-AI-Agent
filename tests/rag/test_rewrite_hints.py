from backend.rag.rewrite import rewrite_question_for_retrieval


def test_rewrite_adds_travel_policy_hints_for_chinese_query():
    question = "我叫 Alice Carter，要去外地出差，根据我的公司信息生成应该注意的 policy"
    retrieval_question = rewrite_question_for_retrieval(question, rewrite_chain=None)
    assert "business travel policy" in retrieval_question
    assert "travel request" in retrieval_question
    assert "company policy" in retrieval_question


def test_rewrite_adds_expense_hints_when_reimbursement_mentioned():
    question = "出差报销流程是什么？"
    retrieval_question = rewrite_question_for_retrieval(question, rewrite_chain=None)
    assert "expense reimbursement policy" in retrieval_question
    assert "approval workflow" in retrieval_question

