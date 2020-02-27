from __future__ import annotations
from typing import Set, Dict, Tuple, Callable
import re

def parse_label_file(filepath : str) -> Tuple[Dict[str, Set[int]],Dict[str, Set[int]],Dict[str, Set[int]]]:
    labelid_to_label = {}
    states_by_label = {}
    labels_by_state = {}

    mark_regexp = re.compile(r"([0-9]+)=\"(.*?)\"")
    line_regexp = re.compile(r"([0-9]+):([\W,0-9]+)")
    with open(filepath) as label_file:
        lines = label_file.readlines()
        regexp_res = re.finditer(mark_regexp, lines[0])
        for _, match in enumerate(regexp_res, start=1):
            labelid, label = int(match.group(1)), match.group(2)
            labelid_to_label[labelid] = label
            states_by_label[label] = set({})

        for line in lines[1:]:
            regexp_res = line_regexp.search(line)
            state_labelids = map(int, regexp_res.group(2).split())
            stateidx = int(regexp_res.group(1))
            labels_by_state[stateidx] = set({})
            for labelid in state_labelids:
                label = labelid_to_label[labelid]
                labels_by_state[stateidx].add(label)
                states_by_label[label].add(stateidx)

    return states_by_label, labels_by_state, labelid_to_label

