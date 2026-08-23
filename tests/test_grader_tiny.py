import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_familiarity_twins as b


def test_answer_head_and_strict_matching():
    assert b.answer_head("England. London is the capital city of England, which is") == "England"
    assert b.answer_head("Debra Hill, however, the movie Clue (1985)") == "Debra Hill"
    assert b.answer_head("Roman Catholic.") == "Roman Catholic"
    # elaboration after the head must not rescue a wrong answer (old whole-answer substring rule did)
    assert b.is_correct("England. London is the capital of England, which is in the United Kingdom", ["Kingdom of Essex", "United Kingdom"]) is False
    assert b.is_correct("England. London is the capital of England, which is in the United Kingdom", ["Kingdom of Essex", "United Kingdom"], strict=False) is True
    # whole-word containment: 'pol' must not match 'Polish'
    assert b.is_correct("a Polish painter", ["politician", "pol"]) is False
    assert b.is_correct("a politician from Minnesota", ["politician", "pol"]) is True
    # exact / prefix / article and punctuation normalization still work
    assert b.is_correct("The Pacific Ocean.", ["Pacific", "Pacific Ocean"]) is True
    assert b.is_correct("Samson Raphaelson and Ernest Vajda.", ["Samson Raphaelson"]) is True
    assert b.is_correct("", ["x"]) is False
