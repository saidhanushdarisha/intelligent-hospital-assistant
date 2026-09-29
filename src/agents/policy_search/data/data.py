import pandas as pd
from langchain_core.documents import Document


def load_csv(file):

    df = pd.read_csv(file)

    documents=[]


    for _,row in df.iterrows():

        text=f"""

ID:
{row.get('policy_id', row.get('guideline_id'))}


Title:
{row.get('title', row.get('guideline_title'))}


Category:
{row.get('category', row.get('condition'))}


Department:
{row['department']}


Description:
{row.get(
'description',
row.get('guideline_description')
)}


Keywords:
{row['keywords']}

"""


        documents.append(
            Document(
                page_content=text,
                metadata={

    "id":
    row.get(
        "policy_id",
        row.get("guideline_id")
    ),

    "department":
    row["department"]
}
            )
        )


    return documents



def load_documents():

    docs=[]

    docs.extend(
        load_csv("src/agents/policy_search/data/policies.csv")
    )

    docs.extend(
        load_csv("src/agents/policy_search/data/guidelines.csv")
    )

    return docs