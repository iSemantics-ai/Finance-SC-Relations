from typing import Dict, List, Tuple
import string
import re
from itertools import chain
from collections import defaultdict
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple


def agg_relations(
    companies_relations,
    org_links,
    initial_item,
    main_relations=["supplier", "customer"],
):
    """
    Aggregates relations for customer and supplier companies from a list of companies_relations.

    @params
    -------
    - companies_relations (dict): A dictionary containing the company name as a key and a list of relations as value.
        Each relation is a dictionary containing the sentence, the relation and the score.
    - main_relations (list, optional): A list of strings representing the main relations to identify.
        Defaults to ['supplier', 'customer'].

    @returns
    --------
    tuple: A tuple containing two lists.
           The first list contains the main relations ('supplier' or 'customer') for each company.
           The second list contains a dictionary with the company name, the sentences and the relation scores.
    """
    relations = []
    relations_attributes = []

    relations_items = []
    for company_name, company_rel in companies_relations.items():
        relation_item = initial_item.copy()

        relation_item["SK"] = {"S": company_name}
        sents = []
        matches_names = org_links[company_name]["matches_names"]
        company_matches = org_links[company_name]["matches"]
        relation_item["extractedNameId"] = (
            {"S": company_matches[0]} if len(company_matches) > 0 else {"S": "null"}
        )
        company_cands = org_links[company_name]["candidates"]
        relation_item["extractedNameCandidateIds"] = (
            {"L": [{"N": str(value)} for value in company_cands]}
            if len(company_cands) > 0
            else {"S": "null"}
        )
        relation_item["representativeName"] = (
            {"S": matches_names[0]} if len(matches_names) > 0 else {"S": "null"}
        )
        relation_item["extractedName"] = {"S": company_name}

        scores = {"supplier": 0, "customer": 0, "other": 0}
        # Define value to aggregate the relation score
        for rel in company_rel:
            sents.append(rel["sentence_id"])
            scores[rel["relation"]] += rel["score"]
            # return sents, scores, company_name
        relation_item["sentenceIds"] = {"L": [{"S": value} for value in sents]}
        company_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]

        relation_item["relationshipLabel"] = {"S": company_score[0]}
        relation_item["predictionScore"] = {"N": str(round(company_score[1], 3))}

        relations_items.append(relation_item)
        # relations.append(company_relation); relations_attributes.append(relation_attribute)
    return relations_items


def match_companies(
    predictions,
    entity_matcher,
    etl_worker,
    lookup_table,
    index_column,
    attribute_name,
    prefix_len,
    sort_len,
    normalized_column,
    id_column,
    database_type="glue",
    cand_thresh=0.8,
    match_thresh=0.98,
    top_k=10,
    index_memory="cpu",
):
    """
    Performs entity matching on a list of company names using the given entity matcher and a lookup table.

    Args:
    predictions (pandas.DataFrame): A DataFrame containing org_groups, which is a dictionary of company names and their
                                     associated probabilities.
    entity_matcher (EntityMatcher): An instance of the EntityMatcher class used to perform entity matching.
    lookup_table (str, optional): The name of the lookup table to use for entity matching. Defaults to 'inferess_companies'.
    index_column (str, optional): The name of the column to use as the index of the lookup table. Defaults to 'tri'.
    threshold (float, optional): The confidence score threshold used to filter out low-confidence matches. Defaults to 0.98.
    top_k (int, optional): The number of top matches to return for each company. Defaults to 10.

    Returns:
    dict: A dictionary containing the matches and candidates for each company.
    """
    # Extract the unique company names from the org_groups column
    all_companies = list(
        set(chain(*predictions.org_groups.apply(lambda x: x.keys()).tolist()))
    )
    all_companies = list(filter(None, all_companies))
    # Generate the trigrams for each company name and extract the unique trigrams

    if database_type == "glue":
        etl.logger.info(
            "Query Glue Table to find the `tri` match companies to search for links"
        )
        companies_tri = list(
            set(
                [
                    re.sub(f"[{re.escape(string.punctuation)}]", "", x)
                    .lower()
                    .replace(" ", "")
                    .replace("the", "")[:prefix_len]
                    for x in all_companies
                ]
            )
        )
        # Load the lookup table with a filter on the trigrams
        inferess_companies = etl_worker.load_athena_with_filter(
            table=lookup_table,
            col=index_column,
            condition="isin",
            value=list(companies_tri),
        )
        etl_worker.logger.info(
            f"Found {inferess_companies.shape[0]} item with the `tri` indecies"
        )
        # Set the index of the DataFrame to normalized_name and build the entity matcher index
        inferess_companies.set_index("normalized_name", inplace=True)

    elif database_type == "dynamodb":
        etl_worker.logger.info(
            f"Query DyanmoDB to find the `{attribute_name}` match companies to search for links"
        )

        queries = []
        for company in all_companies:
            company_prefix = (
                re.sub(f"[{re.escape(string.punctuation)}]", "", company.strip())
                .lower()
                .replace("the", "")
                .replace(" ", "")[:prefix_len]
            )
            if len(company_prefix) == 0:
                continue
            sort = re.sub(
                f"[{re.escape(string.punctuation)}]", "", company.strip()
            ).lower()
            if sort.split(" ")[0] == "the":
                queries.append((company_prefix, sort[: 4 + sort_len].strip()))
                sort = sort[4:].strip()
            queries.append((company_prefix, sort[:sort_len].strip()))

        results = etl_worker.query_dynamodb_table(
            table_name=lookup_table,
            index_name=index_column,
            attribute_name=attribute_name,
            sort_name=normalized_column,
            query_values=queries,
        )
        inferess_companies = pd.DataFrame(results)
        # Set the index of the DataFrame to normalized_name and build the entity matcher index
        etl_worker.logger.info(
            f"Found {inferess_companies.shape[0]} items with prefix lookup"
        )
        inferess_companies.loc[:, "normalized_name"] = inferess_companies[
            normalized_column
        ].apply(lambda x: x["S"])
        inferess_companies.loc[:, "entity_id"] = inferess_companies[id_column].apply(
            lambda x: x["N"]
        )
        inferess_companies.drop_duplicates(
            ["normalized_name", "entity_id"], inplace=True
        )
        inferess_companies.set_index("normalized_name", inplace=True)

    else:
        raise (
            "Error: There must me available database it could be DynamoDb or Glue data catalogue "
        )

    match_results = entity_matcher.match_data(
        source_data=inferess_companies.index.tolist(),
        query_data=all_companies,
        b_size=1000000,
        top_k=top_k,
        threshold=cand_thresh,
        index_memory=index_memory,
    )
    # Initialize a dictionary to store the matches and candidates for each company
    org_links = {}
    # Loop through all_companies and process the matches for each company
    for i, company_name in enumerate(all_companies):
        # Initialize variables to store the matches and candidates for the current company
        matches = []
        candidates = []
        # Get the name of the current  l company
        company_name = all_companies[i]
        # Get the matches for the current company from the results list
        query_output = np.array(match_results.iloc[i]["matches"])
        # Check if there are any matches for the current company
        if query_output.shape[0] > 0:
            # Get the confidence scores for all the matches
            scores = query_output[:, 1].astype(float)
            # Filter out matches with confidence scores below 0.98
            matches_ids = np.where(scores > match_thresh)[0].tolist()
            # If there are any matches with high confidence scores, add them to the matches list
            if matches_ids:
                matches = query_output[matches_ids][:, 0].tolist()
            else:
                matches = []
            # Add all the candidate matches to the candidates list
            candidates = list(
                filter(lambda x: x not in matches, query_output[:, 0].tolist())
            )

        # Add the matches and candidates for the current company to the org_links dictionary
        org_links[company_name] = {
            "matches": inferess_companies.loc[matches]["entity_id"].tolist(),
            "matches_names": matches,
            "candidates": inferess_companies.loc[candidates]["entity_id"].tolist(),
            "candidates_names": candidates,
        }

    return org_links


def process_relations(
    all_relations: pd.DataFrame, matcher, org_links: Dict
) -> List[Dict[str, str]]:
    """
    Group relations by accessionnumber, extract reporter_name mentions and aliases,
    and aggregate relations by company.

    Args:
        all_relations (pandas.DataFrame): DataFrame containing relations to be processed,
            with columns 'accessionnumber', 'org_groups', 'reporter_name', 'aliases', 'relations', and 'sentence'.
        matcher (SimCSE_Matcher): Fuzzy matching object used to match company names.
        org_links (Dict): Results of matching all extracted companies with `company_names` DB.

    Returns:
        List[Dict[str, str]]: A compination of two lists:
            - The first list contains dictionaries representing the relations extracted from each file,
              with keys 'reporterName', 'reporter_cik', 'accessionNumber', 'type', and 'items', and values containing
              information on the relations aggregated by company.
            - The second list contains dictionaries representing the aliases extracted from each file,
              with keys 'PK', 'SK', 'aliasKey', 'aliasTarget', 'type', 'reporterName', and 'accessionNumber', and
              values containing information on the aliases extracted.

    """
    # Initialize lists to store all items and aliases
    all_items = []
    all_aliases = []

    # Group relations by accessionnumber and process each file
    for _, group in all_relations.groupby("accessionnumber"):
        # Extract aliases and create mappings from aliases to names and names to aliases
        aliases = list(filter(None, group["aliases"].tolist()))
        aliases = list(chain(*aliases))
        aliases = set([tuple(l) for l in aliases])
        alias2name = defaultdict(list)
        name2alias = defaultdict(list)
        for k, v in aliases:
            name2alias[k].append(v)
            alias2name[v].append(k)

        # Initialize defaultdict to store relations by company and reporter mentions
        companies_relations = defaultdict(list)

        # Extract reporter mentions and aliases
        all_orgs = np.array(list(set(chain(*group["org_groups"]))))
        # Build index for names and return embeddings for each name
        embs = matcher.build_index(all_orgs.tolist(), return_emb=True)
        # Cluster org names with certain threshold
        results = matcher.search(tuple((all_orgs.tolist(), embs)) ,  threshold=0.96)
        # Initialize basic variables for clustering:
        #   f_list-> flatten_list containes all companies
        #   ids_c-> ids counter to set id for each group
        #   org2id-> organization name mapped to unique id that represent it's group
        #   id2group-> each id mapped to sorted set of groups names
        f_list = []
        ids_c = 0
        org2id = {}
        id2org = {}
        # Loop over the maches to cluster names with cosince similarity
        for c, matches in zip(all_orgs.tolist(), results):
            # Continue if the name existed before
            if c in f_list:
                continue
            # Get all names with high sim score
            n_matches  = [x[0] for x in matches]
            # Add alaises if founded
            n_matches = n_matches +  list(chain(*[alias2name.get(x, []) for x in n_matches]))
            n_matches = n_matches + list(chain(*[name2alias.get(x, []) for x in n_matches]))
            for name in n_matches:
                org2id[name] = ids_c
            # Filter and sort names to set first name is the longest to be representative of the company
            id2org[ids_c] = sorted(set(filter(None, n_matches)), key=lambda x: len(x), reverse=True)
            ids_c += 1
            f_list += n_matches
        # Define all names used for the reporter
        reporter_names = [x[0] for x in matcher.search(group["reporter_name"].iloc[0], threshold=0.96)]
        # Identify reporter mentions and add them to reporter_mentions set
        reporter_mentions = set(
            alias2name.get(group["reporter_name"].iloc[0], [])
            + list(chain(*[name2alias.get(x, []) for x in reporter_names]))
            + [group["reporter_name"].iloc[0]]
            + reporter_names
        )
                # Process each relation in the group
        for _, raw in group.iterrows():
            for rel in raw["relations"]:
                relation = rel.copy()
                is_reporter = False

                # Check if relation mentions reporter and remove reporter mention from relation
                for reporter in reporter_mentions:
                    if relation.get(reporter, None) and relation:
                        is_reporter = True
                        relation.pop(reporter)
                        if not relation:
                            continue
                        score = relation.pop("score")
                        company = list(relation.keys())[0]
                        # Get the representative name of the company
                        representative = id2org[org2id[company]][0]
                        companies_relations[representative].append(
                            {
                                "sentence": raw["sentence"],
                                "sentence_id": raw["sentence_id"],
                                "relation": relation[company],
                                "score": score,
                            }
                        )
        # Aggregate relations by company and append to file_report
        initial_item = {}
        initial_item["PK"] = {"S": f"an#{group['accessionnumber'].iloc[0]}"}
        initial_item["reporterName"] = {"S": group["reporter_name"].iloc[0]}
        initial_item["cik"] = {"S": str(group["reporter_cik"].iloc[0])}
        initial_item["accessionNumber"] = {"S": group["accessionnumber"].iloc[0]}
        initial_item["type"] = {"S": "relationship"}
        initial_item["filingDate"] = {"S": group.iloc[0]["filedasofdate"]}
        
        relations_items = agg_relations(companies_relations, org_links, initial_item)
        all_items += relations_items

        # Create alias items if there are aliases
        if len(alias2name) > 0:
            aliases = {
                "aliases": [
                    {"alias": k, "target": set(v)} for k, v in alias2name.items()
                ],
                "item_info": initial_item,
            }
            all_aliases.append(aliases)

    # Create alias items for each file
    aliases_items = []
    for file_aliases in all_aliases:
        item = file_aliases["item_info"].copy()
        item["PK"] = {"S": f'alias#{item["cik"]["S"]}'}
        item.pop("cik")
        item.pop("reporterName")
        for alias in file_aliases["aliases"]:
            item["aliasKey"] = {"S": alias["alias"]}
            item["SK"] = {"S": f'an#{item["accessionNumber"]["S"]}#{alias["alias"]}'}

            for target in alias["target"]:
                item["aliasTarget"] = {"S": target}
                item["type"] = {"S": "alias"}
                aliases_items.append(item.copy())

    return all_items + aliases_items
