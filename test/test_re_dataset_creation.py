import pytest 
import os 
import sys 
from pathlib import Path
src_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(src_dir)) 
from src.labels_generator.agg_utils import *
from src.labels_generator.data_aggregation import DataAggregator

data_aggregator = DataAggregator(dataset_name= src_dir / 'data/config/llm_aligned_0_1_huge_complex.yaml',
                                 output_dir= src_dir /'data/raw/aggregated_data.json',
                                 entity_matcher= str(src_dir/"artifacts/matcher_model/")
                                 )

@pytest.mark.filterwarnings('ignore::RuntimeWarning')



@pytest.fixture
def datapoint():
    return {
 'filer': 'ADVANCED MICRO DEVICES INC corp',
 'sentence': 'In addition, five customers, including Sony and Microsoft, accounted for approximately 95% of the net revenue attributable to ADVANCED MICRO DEVICES Inc Enterprise, Embedded and Semi Custom segment',
 'relations': [
  ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony'],
  ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft'],
              ],
 'org_groups': {'ADVANCED MICRO DEVICES Inc': 0, 'Microsoft': 1, 'Sony': 2}}


def test_check_relation_tuples():
    assert check_relation_tuples([]) == True
    assert check_relation_tuples([(1, 2, 3)]) == True
    assert check_relation_tuples([(1, 2)]) == False
    assert check_relation_tuples([(1, 2, 3), (4, 5, 6), (7, 8, 9)]) == True

def test_return_possible_pairs():
    assert return_possible_pairs([1, 2, 3]) == [(1, 2), (1, 3), (2, 3)]
    assert return_possible_pairs([]) == []




############################### Test only_filer=true ###############################
def test_only_filer( datapoint):
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= True,
                                max_others= 1) 
    assert llms_relations == [('ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony'),
                              ('ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft')]
    assert other_relations == []

############################### Test only_filer=False & max_other=1 ###############################
def test_onlyfiler_false_max_other1(datapoint):
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= False,
                                max_others= 1) 
    assert llms_relations == [('ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony'),
                              ('ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft')]
    assert other_relations == [('Microsoft', 'other', 'Sony')]

############################### Test Changing Names slightly ###############################
def test_changing_names_slightly(datapoint):
    datapoint['relations'] = [
      ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony Inc'],
      ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft inc'],
                  ]
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= False,
                                max_others= 1) 
    assert llms_relations == [('ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony'),
                              ('ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft')]
    assert other_relations == [('Microsoft', 'other', 'Sony' )]


############################### Test When All Others ###############################
def test_when_all_others(datapoint):
    datapoint['relations'] = [
      ['ADVANCED MICRO DEVICES Inc', 'other', 'Sony Inc'],
      ['ADVANCED MICRO DEVICES Inc', 'other', 'Microsoft inc'],
                  ]
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= False,
                                max_others= 1) 
    assert llms_relations ==[('ADVANCED MICRO DEVICES Inc', 'other', 'Sony'),
     ('ADVANCED MICRO DEVICES Inc', 'other', 'Microsoft')]
    assert other_relations == [('Microsoft', 'other', 'Sony')]

############################### Test When All Others & Only Filer ###############################

def test_when_all_other_only_filer(datapoint):
    datapoint['relations'] = [
      ['ADVANCED MICRO DEVICES Inc', 'other', 'Sony Inc'],
      ['ADVANCED MICRO DEVICES Inc', 'other', 'Microsoft inc'],
                  ]
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= True,
                                max_others= 0) 
    assert llms_relations == [('ADVANCED MICRO DEVICES Inc', 'other', 'Sony'),
     ('ADVANCED MICRO DEVICES Inc', 'other', 'Microsoft')]
    assert other_relations == []

#### Test Adding LLM Relation Not Exist On OrgGroups With Only Filer 
def test_adding_llm_relation_nonexact_onefiler(datapoint):
    datapoint['relations'] = [
      ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony Inc'],
      ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft inc'],
      ['MISTAKE', 'supplier', 'WRONG NAME'],

                  ]
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= False,
                                max_others= 1) 
    assert llms_relations == [('ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony'),
     ('ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft')]

    assert other_relations == [('Microsoft', 'other', 'Sony')]


#########Test Adding LLM Relation Not Exist On OrgGroups Without Only Filer & max_other=2  ###############################

def test_adding_llm_nonexist_max_other_2(datapoint):
    datapoint['sentence'] = 'MISTAKE is supplier WRONG NAME of In addition, five customers, including Sony and Microsoft, accounted for approximately 95% of the net revenue attributable to ADVANCED MICRO DEVICES Inc Enterprise, Embedded and Semi Custom segment'
    datapoint['relations'] = [
      ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony Inc'],
      ['ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft inc'],
      ['MISTAKE', 'supplier', 'WRONG NAME'],
                  ]
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= False,
                                max_others= 2) 
    assert llms_relations == [('ADVANCED MICRO DEVICES Inc', 'supplier', 'Sony'),
     ('ADVANCED MICRO DEVICES Inc', 'supplier', 'Microsoft'),
     ('MISTAKE', 'supplier', 'WRONG NAME')]
    assert len(other_relations) == 2


#########Test having no relation  ###############################

def test_having_no_llm_relations(datapoint):
    datapoint['sentence'] = 'MISTAKE is supplier WRONG NAME of In addition, five customers, including Sony and Microsoft, accounted for approximately 95% of the net revenue attributable to ADVANCED MICRO DEVICES Inc Enterprise, Embedded and Semi Custom segment'
    datapoint['relations'] = []
    llms_relations, other_relations = data_aggregator.extract_relations_from_llm(datapoint,
                                threshold= 0.9 ,
                                only_filer= False,
                                max_others= 2) 
    assert llms_relations == []
    assert len(other_relations) == 2
