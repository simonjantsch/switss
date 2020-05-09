import re
from subprocess import check_output, CalledProcessError
import tempfile
from shutil import copyfileobj


def parse_label_file(filepath):
    """Parses a .lab-file
    
    :param filepath: the filepath
    :type filepath: str
    :return: a dictionary mapping labels to states, a dictionary mapping states to labels, a dictionary mapping indices to labels
    :rtype: Tuple[Dict[str,Set[int]], Dict[int,Set[str]], Dict[int,str]]
    """
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

def prism_to_tra(model_path, destination_path, prism_constants = {}, extra_labels = {}):
    """Translates a prism model into an explicit representation as .tra,.sta,.lab files.
    To this end prism is called, which needs to be present in the path.
    In order to add additional labels to the model, the model file is copied into a temporary file which is appended by the label declarations.
    
    The command that is executed is:

    .. code-block:: sh

        prism filepath -const [C=j for (C,j) in prism_constants] -exportmodel destination_path.tra,sta,lab

    :param model_path: A prism model file (.nm,.pm)
    :type model_path: str
    :param destination_path: a filepath without extension where the resulting explicit model files are written to.
        For example, passing "FILE" will create the files "FILE.tra", "FILE.lab" and "FILE.sta".
    :type destination_path: str
    :param prism_constants: A dictionary of constants to be assigned in the model, defaults to {}
    :type prism_constants: Dict[str, int], optional
    :param extra_labels: A dictionary that defines additional labels (than the ones defined in the prism module) to be added
      	to the .lab file. The keys are label names and the values are PRISM expressions over the module variables. Defaults to {}
    :type extra_labels: Dict[str, str], optional
    :return: True, if model was constructed successfully, otherwise False.
    :rtype: bool
    """    
    const_strings = (["-const"] + [','.join([C+"="+str(i) for (C,i) in prism_constants])]) if len(prism_constants) > 0 else []

    with tempfile.NamedTemporaryFile() as namedtf:
        with open(model_path,"rb") as model_file:
            copyfileobj(model_file,namedtf)

        # write the extra labels to the prism model description
        for (label_name, label_expr) in extra_labels:
            label_string = "label \"" + label_name + "\" = " + label_expr + ";"
            namedtf.write((label_string+"\n").encode("utf-8"))

        namedtf.flush()

        # call prism
        prism_call = ["prism",namedtf.name] + const_strings + ["-exportmodel",destination_path+".tra,sta,lab"]
        try:
            check_output(prism_call)
            return True
        except CalledProcessError as cpe:
            print("Prism call failed.")
            print(' '.join(prism_call))
            print(cpe.stdout.decode("utf-8"))
            return False
