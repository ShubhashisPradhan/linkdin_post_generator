import json
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException 
from llm_helper import llm
import re

def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})
    json_parser = JsonOutputParser()
    try:
        res =  json_parser.parse(response.content)
    except OutputParserException as e:
        #raise OutputParserException(f"Failed to parse output: {response.content}")
        res = {"line_count": None, "language": "Unknown", "tags": []}
    return res

def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    # Loop through each post and extract the tags
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])  # Add the tags to the set

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble
    3. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}
    
    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res

def process_posts(raw_file_path,processed_file_path = None):
    enrichedd_posts = []
    with open(raw_file_path, encoding="utf-8",errors="ignore") as f:
        raw_posts = json.load(f)
        #print(raw_posts)
        for post in raw_posts:
            post['text'] = re.sub(r'[\ud800-\udfff]', '', post['text'])
            meta_data = extract_metadata(post['text'])
            post_with_metadata = post | meta_data
            enrichedd_posts.append(post_with_metadata)

    unified_tag = get_unified_tags(enrichedd_posts)
    #print("Unified Tags:", unified_tag)
    for post in enrichedd_posts:
        current_tags = post['tags']
        new_tags = {unified_tag[tags] for tags in current_tags}
        post['tags'] = list(new_tags)

    with open(processed_file_path, 'w', encoding="utf-8") as outfile:
        json.dump(enrichedd_posts, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process_posts(Path("data\\raw_post.json"), Path("data\\processed_posts.json"))