"""Unit tests for preprocessing.bot_filter."""
from preprocessing.bot_filter import is_bot_author, is_low_quality, should_filter


def test_bot_names_flagged():
    assert is_bot_author("RemindMeBot")
    assert is_bot_author("AutoModerator")
    assert is_bot_author("autotldr")
    assert is_bot_author("repostsleuthbot")


def test_humans_not_flagged():
    assert not is_bot_author("user_4321")
    assert not is_bot_author("ronaldofan")


def test_short_text_low_quality():
    assert is_low_quality("ok")
    assert is_low_quality("")


def test_promo_text_low_quality():
    assert is_low_quality("Use my promo code GOAL2026 for a bonus on this match!")


def test_all_caps_low_quality():
    assert is_low_quality("WHY ARE THEY BOTTLING IT AGAIN I CANT BELIEVE THIS")


def test_normal_text_passes():
    assert not is_low_quality("England looked sharp today, Bellingham was excellent.")


def test_combined_filter():
    assert should_filter("AutoModerator", "Anything")
    assert not should_filter("real_fan", "Solid performance by the team in the second half.")
