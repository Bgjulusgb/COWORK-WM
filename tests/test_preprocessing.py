from preprocessing.pipeline import PreprocessingPipeline


def test_negation_preserved_through_stopword_removal():
    p = PreprocessingPipeline(home_keywords=["England"], away_keywords=["Croatia"])
    out = p.process("England was not good today")
    assert "not" in out.tokens, f"negation lost: {out.tokens}"


def test_slang_expansion():
    slang = {"W": "win", "clean sheet": "no goals conceded"}
    p = PreprocessingPipeline(slang_dict=slang, home_keywords=["England"], away_keywords=["Croatia"])
    out = p.process("England got the W today, clean sheet!")
    assert "win" in out.cleaned
    assert "no goals conceded" in out.cleaned


def test_team_attribution_home():
    p = PreprocessingPipeline(home_keywords=["England", "Kane"], away_keywords=["Croatia"])
    out = p.process("Kane is on fire today!")
    assert out.team_attribution == "home"


def test_team_attribution_neutral_when_both():
    p = PreprocessingPipeline(home_keywords=["England"], away_keywords=["Croatia"])
    out = p.process("England vs Croatia tonight is going to be wild")
    assert out.team_attribution == "neutral"


def test_engagement_weight_monotonic_in_score():
    p = PreprocessingPipeline()
    w_low = p.process("a", score=1, num_comments=1, upvote_ratio=0.9).engagement_weight
    w_high = p.process("a", score=1000, num_comments=1, upvote_ratio=0.9).engagement_weight
    assert w_high > w_low
