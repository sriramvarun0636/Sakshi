import numpy as np
import pandas as pd
import pytest

from sakshi.baselines.clip_zeroshot import CLASSES, PROMPTS, classify_embeddings


def test_clip_zeroshot_prompt_coverage_and_hierarchical_predictions():
    assert all(set(PROMPTS[head]) == set(classes) for head, classes in CLASSES.items())
    vectors = {}
    for head, classes in CLASSES.items():
        array = np.zeros((len(classes), 512), dtype=np.float32)
        array[:, 1] = 1
        array[0, 0] = 1
        array[0, 1] = 0
        vectors[head] = array
    # Informative class is index one in the canonical label order.
    vectors["informative"][0] = 0
    vectors["informative"][0, 1] = 1
    vectors["informative"][1] = 0
    vectors["informative"][1, 0] = 1
    image = np.zeros((2, 512), dtype=np.float32)
    image[0, 0] = 1
    image[1, 1] = 1
    output = classify_embeddings(["informative", "ordinary"], image, vectors)
    assert output.informative_label.tolist() == ["informative", "not_informative"]
    assert output.humanitarian_label.iloc[1] == "not_humanitarian"
    assert pd.isna(output.severity_label.iloc[1])
    assert output.alpha_image.tolist() == [1.0, 1.0]
    for head, classes in CLASSES.items():
        columns = [f"p_{head}__{name}" for name in classes]
        np.testing.assert_allclose(output[columns].sum(axis=1), 1, atol=1e-6)

    ambiguous_vectors = {head: np.ones((len(classes), 512), dtype=np.float32) for head, classes in CLASSES.items()}
    ambiguous = classify_embeddings(["half"], image[:1], ambiguous_vectors, logit_scale=1)
    assert ambiguous.informative_label.iloc[0] == "informative"
    assert ambiguous.humanitarian_label.iloc[0] != "not_humanitarian"
    assert ambiguous["p_humanitarian__not_humanitarian"].iloc[0] == pytest.approx(0.5)


def test_clip_zeroshot_rejects_misaligned_features():
    vectors = {head: np.ones((len(classes), 512), dtype=np.float32) for head, classes in CLASSES.items()}
    image = np.zeros((1, 512), dtype=np.float32)
    image[0, 0] = 1
    with pytest.raises(ValueError, match="unique sample IDs"):
        classify_embeddings(["same", "same"], np.vstack((image, image)), vectors)
    with pytest.raises(ValueError, match="unit vectors"):
        classify_embeddings(["one"], image * 2, vectors)
