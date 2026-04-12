import pytest
from src.orchestrator.analyzer import TaskAnalyzer
from src.types import Domain, Complexity


def test_classify_math_question():
    analyzer = TaskAnalyzer()
    result = analyzer.classify("What is the derivative of x^2 + 3x?")
    assert result.domain == Domain.MATH


def test_classify_code_question():
    analyzer = TaskAnalyzer()
    result = analyzer.classify("Write a Python function to reverse a linked list")
    assert result.domain == Domain.CODE


def test_classify_general_question():
    analyzer = TaskAnalyzer()
    result = analyzer.classify("What is the capital of France?")
    assert result.domain == Domain.GENERAL


def test_classify_reasoning_question():
    analyzer = TaskAnalyzer()
    result = analyzer.classify(
        "If all roses are flowers and some flowers fade quickly, "
        "can we conclude that some roses fade quickly?"
    )
    assert result.domain == Domain.REASONING


def test_classify_simple_complexity():
    analyzer = TaskAnalyzer()
    result = analyzer.classify("What is 2 + 2?")
    assert result.complexity == Complexity.SIMPLE


def test_classify_with_mmlu_subject_hint():
    analyzer = TaskAnalyzer()
    result = analyzer.classify(
        "Which vitamin is fat-soluble?",
        subject_hint="anatomy",
    )
    assert result.domain == Domain.GENERAL


def test_classify_with_math_subject_hint():
    analyzer = TaskAnalyzer()
    result = analyzer.classify(
        "Find the eigenvalues of the matrix [[2,1],[1,2]]",
        subject_hint="abstract_algebra",
    )
    assert result.domain == Domain.MATH


def test_classify_returns_confidence():
    analyzer = TaskAnalyzer()
    result = analyzer.classify("Solve: 3x + 7 = 22")
    assert 0.0 <= result.confidence <= 1.0
