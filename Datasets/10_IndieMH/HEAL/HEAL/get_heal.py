
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L12-v2',device="cuda:1")

import json
filename = './nodes/affective_states.txt'
with open(filename) as f:
    affective_states = f.read()
    affective_states = json.loads(affective_states)
    # #print(affective_states)
# #print(affective_states["nodes"][0])
# #print(len(affective_states["nodes"]))


filename = './nodes/expectations.txt'
with open(filename) as f:
    expectations = f.read()
    expectations = json.loads(expectations)
    # #print(expectations)
#print(expectations["nodes"][0])


filename = "./nodes/feedback.txt"
with open(filename) as f:
    feedback = f.read()
    feedback = json.loads(feedback)
    # #print(feedback)
#print(feedback["nodes"][0])


filename = "./nodes/responses.txt"
with open(filename) as f:
    responses = f.read()
    responses = json.loads(responses)
    #print(responses)
#print(responses["nodes"][0])

filename = "./nodes/stressors.txt"
with open(filename) as f:
    stressors = f.read()
    stressors = json.loads(stressors)
    # #print(stressors)
#print(stressors["nodes"][0])


filename = "./edges/affective_states.txt"
with open(filename) as f:
    affective_states_edges_s = f.read()
    affective_states_edges_s = json.loads(affective_states_edges_s)
    # #print(affective_states_edges_s)
# #print(affective_states_edges_s["edges"][0])
# from stressors to affective states


filename = "./edges/expectations-responses.txt"
with open(filename) as f:
    expectations_edges_r = f.read()
    expectations_edges_r = json.loads(expectations_edges_r)
    # #print(expectations_edges_r)
# #print(expectations_edges_r["edges"][0])
# from expectations to responses


filename = "./edges/expectations-stressors.txt"
with open(filename) as f:
    expectations_edges_s = f.read()
    expectations_edges_s = json.loads(expectations_edges_s)
    # #print(expectations_edges_s)
# #print(expectations_edges_s["edges"][0])
# from stressors to expectations

filename = "./edges/responses-feedback.txt"
with open(filename) as f:
    responses_edges_f = f.read()
    responses_edges_f = json.loads(responses_edges_f)
    # #print(responses_edges_f)
# #print(responses_edges_f["edges"][0])
# from responses to feedback

filename = "./edges/stressors-responses.txt"
with open(filename) as f:
    stressors_edges_r = f.read()
    stressors_edges_r = json.loads(stressors_edges_r)
    # #print(stressors_edges_r)
# #print(stressors_edges_r["edges"][0])
# from stressors to responses

def get_nodes(input,all_nodes,k):
    sentences = []
    for i in range(len(all_nodes)):
        sentences.append(all_nodes[i]["label"])
    embedding_input = model.encode(input, convert_to_tensor=False)
    embeddings = model.encode(sentences, convert_to_tensor=False)
    # #print(embeddings.shape)
    # #print(embedding_input.shape)
    cosine_scores = util.pytorch_cos_sim(embedding_input, embeddings)
    # #print(cosine_scores)
    # #print(cosine_scores[0][0].item())
    for i in range(len(all_nodes)):
        all_nodes[i]["score"] = cosine_scores[0][i].item()
    all_nodes.sort(key=lambda x: x["score"], reverse=True)
    return all_nodes[0:k]

def find_edges(node,edges,check_nodes):
    final_node = None
    simialirty_score = 0
    for i in range(len(edges)):
        if edges[i]["from"] == node["id"] or edges[i]["to"] == node["id"]:
            to_id = edges[i]["to"]
            from_id = edges[i]["from"]
            # #print(to_id)
            for j in range(len(check_nodes)):
                if check_nodes[j]["id"] == to_id or check_nodes[j]["id"] == from_id:
                    if check_nodes[j]["score"] > simialirty_score:
                        simialirty_score = check_nodes[j]["score"]
                        final_node = check_nodes[j]
    return final_node

path = "../../New_Corrected_data/Patient_Only_Context/"

import json
with open(path+"train_erc_instances_with_commonsense_final.json","r") as f:
    train = json.load(f)

with open(path+"test_erc_instances_with_commonsense_final.json","r") as f:
    test = json.load(f)

with open(path+"val_erc_instances_with_commonsense_final.json","r") as f:
    val = json.load(f)

import tqdm
from tqdm import tqdm

def update_data(train):
    for key in tqdm(train.keys()):
        train[key]["heal"] = {}
        try:
            if "commonsense" not in train[key]:
                continue
            topics = train[key]["topics"]
            topic = " ".join(topics)
            if len(topic) > 50:
                topic = topic[0:50]
            input = topic
            commonsense = train[key]["commonsense"]
            x_react = commonsense["xReact"]
            x_want = commonsense["xWant"]
            x_need = commonsense["xNeed"]
            x_intent = commonsense["xIntent"]
            x_effect = commonsense["xEffect"]
            print(input)
            # print(x_react["beams"][0])
            # print(x_want["beams"][0])
            # print(x_need["beams"][0])
            # print(x_intent["beams"][0])
            # print(x_effect["beams"][0])
            stressor_node_x = None
            # print(x_intent["beams"])
            affective = None
            want = None
            need = None
            effect = None
            for j in range(3):
                if x_intent["beams"][j].lower() != "none" and stressor_node_x == None:
                    stressor_node_x = x_intent["beams"][j]
                if x_react["beams"][j].lower() != "none" and affective == None:
                    affective = x_react["beams"][j]
                if x_want["beams"][j].lower() != "none" and want == None:
                    want = x_want["beams"][j]
                if x_need["beams"][j].lower() != "none" and need == None:
                    need = x_need["beams"][j]
                if x_effect["beams"][j].lower() != "none" and effect == None:
                    effect = x_effect["beams"][j] 
            print(stressor_node_x,affective,need,want,effect)
            if stressor_node_x == None or affective == None or (want is None and need is None and effect is None):
                # train["expectation"].append({})
                # train["stressor"].append({})
                # train["affective_state"].append({})
                # train["response"].append({})
                print("adding no knowledge")
                train[key]["heal"]["expectation"] = {}
                train[key]["heal"]["stressor"] = {}
                train[key]["heal"]["affective_state"] = {}
                train[key]["heal"]["response"] = {}
            else:
                stressor_node = stressor_node_x
                expectation_nodes = get_nodes(input,expectations["nodes"],1)
                stressor_nodes = get_nodes(stressor_node,stressors["nodes"],1)
                affective_state_nodes = get_nodes(x_react["beams"][0],affective_states["nodes"],1)
                if want == None:
                    want = "none"
                if need == None:
                    need = "none"
                if effect == None:
                    effect = "none"
                response_x = str(want) + " , " + str(need) + " , " + str(effect)
                response_nodes = get_nodes(response_x,responses["nodes"],1)
                print("fetched nodes")
                stressor_node = stressor_nodes[0]
                expectations_connection = find_edges(stressor_node,expectations_edges_s["edges"],expectations["nodes"])
                response_connection = find_edges(stressor_node,stressors_edges_r["edges"],responses["nodes"])
                affective_state_connection = find_edges(stressor_node,affective_states_edges_s["edges"],affective_states["nodes"])
                # train["expectation"].append(expectations_connection)
                # train["stressor"].append(stressor_node)
                # train["affective_state"].append(affective_state_connection)
                # train["response"].append(response_connection)
                train[key]["heal"]["expectation"] = expectations_connection
                train[key]["heal"]["stressor"] = stressor_node
                train[key]["heal"]["affective_state"] = affective_state_connection
                train[key]["heal"]["response"] = response_connection
        except Exception as e:
            print(e)
    return train

new_train = update_data(train)

with open(path+"train_erc_instances_with_commonsense_final_heal.json","w") as f:
    json.dump(new_train,f)

new_test = update_data(test)

with open(path+"test_erc_instances_with_commonsense_final_heal.json","w") as f:
    json.dump(new_test,f)

new_val = update_data(val)

with open(path+"val_erc_instances_with_commonsense_final_heal.json","w") as f:
    json.dump(new_val,f)

