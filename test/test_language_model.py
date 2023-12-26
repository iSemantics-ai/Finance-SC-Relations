import pytest 
import os 
import sys 
from pathlib import Path
src_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(src_dir)) 
from src.language_model.spacy_loader import SpacyLoader
@pytest.mark.filterwarnings('ignore::RuntimeWarning') # notice the ::

@pytest.fixture
def spacy_loader():
    spacy_loader = SpacyLoader(lm='en_core_web_trf',
            require_gpu=True,
            load_matcher=True)
    return spacy_loader

@pytest.fixture
def check_data():
    return ['Hewlett-Packard reports that "Cisco", NetApp Inc, Lenovo Group Ltd, International Business Machines Corporation "IBM", Huawei Technologies Co Ltd, Amazon.com Inc "Amazon", Oracle Corporation or "Oracle", Fujitsu Limited ("Fujitsu"), Juniper Networks Inc, Inspur Co, Ltd, Hitachi Ltd, Extreme Networks Inc, Pure Storage Inc, Brocade Communications Systems Inc, VMware, Nutanix Inc, Google Inc and Rackspace Inc']

@pytest.fixture
def expected_aliases():
    return [[('International Business Machines Corporation', 'IBM'),
                    ('Amazon.com Inc', 'Amazon'),
                    ('Fujitsu Limited', 'Fujitsu')],
                  ]
def test_org_grouping(spacy_loader, check_data, expected_aliases):
    sents, spans , groups, aliases = spacy_loader.predictor(check_data) 
    for i, alias_group in enumerate(aliases):
        for target, alias in alias_group:
            assert (target, alias) in expected_aliases[i]
            assert groups[i][target] == groups[i][alias]
