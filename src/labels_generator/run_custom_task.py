import os
import sys
import argparse
import pandas as pd
import openai
import pandas as pd
from glob import glob
from tqdm import tqdm
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KE")

current_path = Path.cwd()
src_dir = current_path
print("src_dir", src_dir)
sys.path.append(str(src_dir))
# import annotation methods
from src.labels_generator import (generate_relations_with_explanation, create_sorted_relation,
                                  relations_tupled,relations_tupled_2, deserialize_json_dict2)

# Load matcher
from src.matcher.core import SimCSE_Matcher
matcher = SimCSE_Matcher(str(src_dir/ 'artifacts/matcher_model'))


replaces = {"sentence": "{sentence}"}
# Replace the keys with values for unified relation direction
relations_map = {"customer": "supplier"}

Explanation_prompt  ='''
Your task is to provide an explanation about the relation between companies and role they represent in the relation from the report given in ``` quote.
​
- possible relation - {supplier_and_customer, financial_trade, nothing}
- role - {supplier, customer} 
- role is required for supplier_and_customer relation only, Identifying which company is supplier and which company is customer is important.
​
##Report
```
{sentence}
​
```
​
Here are some definitions that might help to understand how to identify the relation between companies
​
## Rules for `financial_trade`
- Companies engaged in providing financing or investing in another company are in a financial_trade relation
- If companies are involved in the buying and selling of shares or ownership interests in each other, they are in a financial_trade relation.
- When companies are mentioned in the context of mergers and acquisitions, they have a financial_trade relation.
- Companies participating in the purchase of assets or transactions involving related parties are in a financial_trade relation.
- If companies have a debtor and creditor relationship with each other, they have financial_trade relation.
- Companies conducting transactions for the purpose of managing their working capital have a financial_trade relation.
- Companies engaged in any form of payment, excluding revenue or sales contributions, with another company have financial_trade relation.
​
​
## Rules for identifying the `supplier_and_customer` relation and assigning role of the customer or supplier  - 
- Customer companies may be referred as contributing to the revenue stream of supplier companies
- Customer companies may be referred as companies accounting for revenues or net sales of supplier companies
- Customer companies may be referred as companies contributing certain percentage of revenues or net sales of supplier companies in certain year.
- Customer companies are accountable for outstanding payment in trade balance, accounts receivable or amount payable to supplier companies
- Customer companies may be referred to as companies that assign or transfer manufacturing responsibilities to suppliers
- Customer companies may be referred to as those that rely on supplier companies for key inputs or materials required in their production process.
- Customer companies may be referred to as companies that purchase some product, goods or service from supplier
- Customer companies may be referred to as distributors, users, commercializers, retailers, insurance companies, or similar entities.
- Customer companies could act as a distribution channel. Supplier has original content, and supplier distribute it via customer companies platform or channel
- Customer company may be referred as paying royalty to another company
- Customer company may be referred as using supplier company's technology or service 
- In collaboration agreement or joint development, it should be clear that customer company will contribute to the supplier companies revenue
- Supplier companies depend on customer companies to supply or sale of their product
- Supplier companies may be referred to as companies that gain or take revenues from customers
- Supplier companies may be referred as companies whose revenue is increased or decreased or have revenue impact due to customer companies
- Supplier companies receive outstanding payments in trade balance, accounts receivable or amount payable from customer companies
- Supplier companies may be referred to as those who own, operate, or manufacture materials and then sell to customer companies
- Supplier companies may be referred to as vendors or providers of services, products, or materials to customer companies
- Supplier companies may be referred to as giving license to customer company. licensor would be supplier and licensee would be customer company.
- Supplier companies may be referred to as entities that offer discounts, special pricing to customer companies due to their business relationship.
- Supplier companies may say there is `no outstanding receivable balance or payment` from the customer company when all payments are completed. 
- As per rules, if one company is identified as customer then other company who is doing business with the customer company becomes supplier
- As per rules, if one company is identified as supplier then other company who is doing business with the supplier company becomes customer
- Always remember `supplier_and_customer` has two roles, so you need to identify supplier company name and customer company name.
​
​
## Complex rules for `financial_trade`, `supplier_and_customer` relation
- If two companies are involved in a collaboration agreement or joint development, then think in the following steps -
    1. find which company is paying money to another company in joint development or agreement
    2. if it is not clear who is paying money, then the relation is financial_trade
    3. if it is clear who is payee, the payee is the customer
​
## Rules to follow for creating output -
- Use only the information provided in the report
- Do not infer anything outside of the given context
- Check all rules, and think step by step as mentioned for complex relations
- Find the correct relation by applying given rules of all relations
- Write the explanation for only indentified relation as per below instructions
- Skip the explanation for unindentified relation, when neither `supplier_and_customer` nor `financial_trade` relation is found, then mention nothing relation
- Explanation when identified relation is `supplier_and_customer` 
   - Mention the rule that has matched to identify the relation
   - Write short explanation on how rule is matched to the report
   - As supplier_and_customer relation has two roles, mention which company has which role and write a statement about that
   - Always mention name of customer and name of supplier company saying who is supplier and who is customer
   - If more than one customers are identified, then mention all customer company names with respect to supplier company name
   - If more than one supplier are identified, then mention all supplier company names with respect to customer company name 
   - Mention all company names in the report that are part of relation
   - Mention the rule that has matched to identify the relation
- Explanation when identified relation is `financial_trade` or `Nothing`
    - Mention the rule that has matched to identify the `financial_trade` or `Nothing` relation
    - Write short explanation on how rule is matched to the report
    - Write all company names that are in `financial_trade` or `Nothing` relation
- Conclude your answer, making sure it is related to the defined rules only
'''

Relation_prompt= '''
Your task is to identify the relation and role of company mentioned in the explanation given in ``` quote 
​
- possible relation - {supplier_and_customer, financial_trade, nothing}
- role - {supplier, customer} 
- role would be mentioned only for supplier_and_customer relation
​
##Explanation
```
{explanation}
​
```
​
## Rules to follow for creating output -
- use only information given in the explanation
- Don't infer anything outside of the given explanation
- for `supplier_and_customer` relation 
     - examine the explanation to find the mentioned customer and supplier company names
     - supplier_and_customer relation holds value only when the identities of the customer and supplier are specified
     - For all supplier and customer pairs, assign customer company name to assign supplier company name
     - As per explanation, find all correct customer and supplier company pairs and add it to the output json
- for `financial_trade` and `Nothing` relation 
     - find all companies mentioned for this relation
     - Create pairs of company names. A pair has two company names in it.
- Output should be strictly in JSON Object for the companies mentioned in the explanation
- If the explanation `supplier_and_customer` relation, then map company names to their role in the output:
    - If company is customer in relation, then it's name is mapped to customer key
    - If company is supplier in relation, then it's name is mapped to supplier key
    - role customer and supplier are keys in json and the values are respective company names.
    - example -  {'supplier_and_customer' : [ {'customer': 'company_name_acting_as_customer', 'supplier': 'company_name_acting_as_supplier'} ] } 
- If two companies are in financial_trade relation, then create list of all such two companies involved in the financial_trade relation
     - example -  { 'financial_trade': [ [ company1_name, company2_name ] ] }
- If two companies are in nothing relation, then create a list of all such two companies involved in the nothing relation
     - example -  { 'nothing': [ [ company1_name, company2_name ] ] } 
- Don't replicate relations
- Output should be strict json object that can be parsed
​
​
## output
Return single JSON Object that contains relations with the following format in ``` quote -
```{'supplier_and_customer' : [ {'customer': 'company_acting_as_customer', 'supplier': 'company_acting_as_supplier'} ],
   'financial_trade': [ [company1_name, company2_name] , [company1_name, company2_name] ], 
   'nothing': [ [company1_name, company2_name] , [company1_name, company2_name]]}```
​'''
# Generate list of relations in tuples
explantion_replaces = {"sentence": "{sentence}"}
relation_replaces = {"explanation": "{explanation}"}
# Replace the keys with values for unified relation direction
relations_map = {"customer": "supplier"}
# Generate list of relations in tuples

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='GPT Annotation Job arguments')
    parser.add_argument('--batch', nargs='+', type=int, help='Discribe the batch to be annotated')

    # Define the arguments you want to accept
    # Parse the command-line arguments
    args = parser.parse_args()

    batch = args.batch
    print("batch", batch)
    data  = pd.read_excel("data/tasks/finetune_llm_on_label_1/source_data.xlsx", index_col="index")
    tqdm.pandas(desc="Create sme_relations column")
    data['sme_relations']=data[['entity_2', 'inf_relations', 'entity_1','Label']]\
        .progress_apply(lambda x:\
        create_sorted_relation(x[0],
            x[1],
            x[2],
            x[3],
            relations_map=relations_map),
            axis = 1)
    
    out_file = f"data/tasks/finetune_llm_on_label_1/llm_relations_with_explained_v2.2_{batch[0]}_{batch[1]}"
    print("out_file", out_file)
    output = generate_relations_with_explanation(data=data[batch[0]:batch[1]],
                        explanation_prompt=Explanation_prompt,
                        relation_prompt=Relation_prompt,
                        explantion_replaces=explantion_replaces,
                        relation_replaces=relation_replaces,
                        relations_map=relations_map,
                        deserialize_func=deserialize_json_dict2,
                        tuple_func=relations_tupled_2,
                        do_explain=True,
                        do_relation=True)
    print("output.shape", output.shape)
    print("output.columns", output.columns)

    output.to_excel(f"{out_file}.xlsx", index_label="index")