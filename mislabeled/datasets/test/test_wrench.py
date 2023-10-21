import os
from tempfile import TemporaryDirectory

from ..wrench import fetch_wrench

# @pytest.mark.parametrize("name", WRENCH_DATASETS.keys())
# def test_fetch(name):
#     dataset = fetch_wrench(name)
#     assert "data" in dataset
#     assert "target" in dataset
#     assert "weak_targets" in dataset
#     assert "target_names" in dataset
#     assert "description" in dataset


def test_youtube():
    with TemporaryDirectory() as tmpdir:
        youtube = fetch_wrench("youtube", cache_folder=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "youtube"))

    assert len(youtube["data"]) == 1686
    assert isinstance(youtube["data"][0], str)
    assert len(youtube["target"]) == 1686
    assert len(youtube["weak_targets"]) == 1686
    assert len(youtube["weak_targets"][0]) == 10
    assert youtube["target_names"] == ["HAM", "SPAM"]
