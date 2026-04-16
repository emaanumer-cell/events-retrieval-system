RETRIEVAL SYSTEM:

STEP 1:
1) Frontend is structured as one section: Theres an input box for query and a filter icon to filter by screen if needed. Below the query should be some helpful example queries to write, listing possible metric names to include
2) Get input from user structured as {"query":..., "screens":..}
3) use an LLM call to expand on the query keywords. so e.g if acquisition is written, we want to capture as many events that coudl fall under acquisition so the llm should list common keywords that might be used in apps. (gemini 3 pro-preview, 0.5 temp, max output token 256)
4) In my env, there are pinecone records. example of a record is as follows.I If there is a filter by screen name then only fetch with matching screen name. Otherwise fetch all of them:

    ID: 04a01b3dbd922928

values: []

metadata:

event_name: "creating_screen_shown"

full_event_json: "{\"event_name\": \"creating_screen_shown\", \"event_definition\": \"Fired when the AI Creating Animation loading screen is displayed.\", \"screen_name\": \"AI Creating Animation\", \"detailed_event_definition\": \"This event fires when the AI Creating Animation loading screen appears with progress steps. It means the backend has started processing the animation request. This answers: How many animations enter the generation pipeline? What is the average time from this event to completion or failure? It affects the user by showing them a progress indicator while they wait and affects the company by representing a committed server resource (API call cost). Upstream, the user tapped animate (and possibly watched an ad); downstream, the animation will either succeed (preview) or fail (error). Significant for monitoring backend health and generation success rates. Example use case: Measure time between creating_screen_shown and either anim_preview_screen_shown or anim_gen_error_shown to calculate average generation time and failure rate.\", \"key_event\": \"No\", \"parameters\": [{\"name\": \"image_id\\nanimation_style\\nduration_seconds\", \"description\": \"ID of the image being animated\\nStyle of animation if selected\\nDuration of the animation being generated\", \"sample_values\": \"img_abc123\\nwiggle, blink_head, cartoon_bounce, zoom_pan\\n4, 6, 8\"}]}"

key_event: "No"

screen_name: "AI Creating Animation"


5) Here i want to test two options. i want to check retrieval quality with a bi-encoder and also with a cross-encoder. For a cross-encoder, directly compare the full row with the query, compute the score, carry out min-max on it, and display. dont put any threshold to filter out anything. for now i want to see the full score of each event and how it performs accordingly.
For a bi-encoder, embed the query and compare with the embeddings fetched from pinecone above. implement hybrid retrieval in this case. also display the values on the frontend under bi-encoder section. use same model for embedding the query that is done in code for indexing. Return everything so theres score for me to view.
basically im trying to see how bi-encoders vs cross-encoders perform

6) currently, fast api is implemented but i want you to keep it simple and use streamlit instead. 


The above requires serious re-writing and refactoring of already made code. I would suggest you prioritize the indexing part of the old code becasue that is kept the same but do not study the one you have to replace with abvoe. make a logs folder which rewrites into one file the logs of the most recent run so i can debug any issue that might happen.