from typing import Dict, List, Tuple

import numpy as np


def unique_people(labels: np.ndarray) -> np.ndarray:
    """Return sorted unique person ids."""
    return np.unique(labels)


def indices_for_person(labels: np.ndarray, person_id: int) -> np.ndarray:
    """Return indices belonging to a person id."""
    return np.where(labels == person_id)[0]


def default_selection(labels: np.ndarray) -> Tuple[int, int]:
    """Pick first person and first index as defaults."""
    people = unique_people(labels)
    person = int(people[0])
    idx = int(indices_for_person(labels, person)[0])
    return person, idx


def person_display_name(person_id: int) -> str:
    return f"Person {person_id}"


def selection_caption(person_id: int, index: int) -> str:
    return f"Person: {person_display_name(person_id)} (index {index})"


def template_mean_by_person(z: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute mean PCA vector per person.

    Args:
        z: projected data (N, k)
        labels: person ids (N,)
    """
    templates: Dict[int, np.ndarray] = {}
    for pid in unique_people(labels):
        mask = labels == pid
        templates[int(pid)] = z[mask].mean(axis=0)
    return templates
