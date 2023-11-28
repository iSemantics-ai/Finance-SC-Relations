import pytest 
import os 
import sys 
from pathlib import Path
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KE")
src_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(src_dir)) 
from src.labels_generator import (relation_search,resort_relation)


@pytest.fixture
def matcher():
    # Load matcher
    from src.matcher.core import SimCSE_Matcher
    matcher = SimCSE_Matcher('artifacts/matcher_model')
    return matcher

@pytest.fixture
def relations_map():
    return {"customer":"supplier"}

def test_resort_tuple(relations_map):
    assert resort_relation(("Apple inc", "customer", "Google"), relations_map)\
        ==['Google', 'supplier', 'Apple inc']

def test_relation_search(matcher,relations_map):
    relations = [
        ['ORBCOMM Inc', 'nothing', 'Systems Inc'],
        ['ORBCOMM Inc', 'nothing', 'inthinc Technology Solutions Inc'],
        ['ORBCOMM Inc', 'nothing', 'Value added Solutions Providers'],
        ['Onixsat', 'supplier', 'ORBCOMM Inc'],
        ['Satlink', 'supplier', 'ORBCOMM Inc'],
        ['Sascar', 'supplier', 'ORBCOMM Inc'],
        ['Carrier Transicold', 'supplier', 'ORBCOMM Inc'],
        ['Thermo King', 'supplier', 'ORBCOMM Inc'],
        ['CS Wholesale', 'supplier', 'ORBCOMM Inc'],
        ['Canadian National Railways', 'supplier', 'ORBCOMM Inc'],
        ['CR England', 'supplier', 'ORBCOMM Inc'],
        ['Hub Group Inc', 'supplier', 'ORBCOMM Inc'],
        ['KLLM Transport Services', 'supplier', 'ORBCOMM Inc'],
        ['Marten Transport', 'supplier', 'ORBCOMM Inc']]
    relations_test = [
     [('ORBCOMM Inc', 'nothing', 'inthinc Technology Solutions Inc'), True], 
     [('Satlink', 'nothing', 'inthinc Technology Solutions Inc'), True],
     [('Onixsat', 'supplier', 'ORBCOMM'), True],
     [('ORBCOMM Inc', 'supplier', 'Systems Inc'), False],
     [('Onixsat', 'nothing', 'ORBCOMM Inc'), False],
     [('ORBCOMM Inc', 'nothing', 'Onixsat'), False],
     [('ORBCOMM Inc', 'supplier', 'Onixsat'), False],
     [('Random1', 'nothing', 'Random2'), True],
     [('Hub Group', 'supplier', 'ORBCOMM Inc'), True],
     [('Sascar', 'supplier', 'Onixsat'), False],
     [('Sascar', 'nothing', 'Onixsat'), True]]
    for relation_test in relations_test:
        expected = relation_test[1]
        answer = relation_search(
        query_relation=relation_test[0],
        relations_tuples=relations,
        matcher=matcher,
        threshold=0.85,
        main_relations=list(relations_map.values())
        )
        if answer != expected:
            print("relation_test", relation_test[0])
            print("relations", relations)
            print("expected", expected)

        assert answer == expected

    


    
