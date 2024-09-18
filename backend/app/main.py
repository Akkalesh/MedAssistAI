from fastapi import FastAPI
from transformers import T5ForConditionalGeneration, AutoTokenizer
import uvicorn

app = FastAPI()

# Load fine-tuned model and tokenizer
model_name = "/home/project/BE/models/lora_fine_tuned_model_medical-qna-3k"
model = T5ForConditionalGeneration.from_pretrained(model_name) 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Enhanced Chain-of-Thought (CoT) Prompting
def cot_prompt(text):
    return (
        "Let's analyze the input step by step."
        "First, identify the main problem mentioned. Then, consider possible causes and suggest treatments. "
        f"Now, based on this patient's symptoms: {text}, what could be the cause and what should be the treatment? Only give me the docktor's part as response."
        # f"{text}"
    )

# Enhanced Few-Shot Examples with Medical Queries and Responses
def few_shot_prompt():
    examples = (
        "Is D-bifunctional protein deficiency inherited ?.\n"
        "This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, but they typically do not show signs and symptoms of the condition.\n\n"
        "What is (are) Tourette syndrome ?\n"
        "Tourette syndrome is a complex disorder characterized by repetitive, sudden, and involuntary movements or noises called tics. Tics usually appear in childhood, and their severity varies over time. In most cases, tics become milder and less frequent in late adolescence and adulthood. Tourette syndrome involves both motor tics, which are uncontrolled body movements, and vocal or phonic tics, which are outbursts of sound. Some motor tics are simple and involve only one muscle group. Simple motor tics, such as rapid eye blinking, shoulder shrugging, or nose twitching, are usually the first signs of Tourette syndrome. Motor tics also can be complex (involving multiple muscle groups), such as jumping, kicking, hopping, or spinning. Vocal tics, which generally appear later than motor tics, also can be simple or complex. Simple vocal tics include grunting, sniffing, and throat-clearing. More complex vocalizations include repeating the words of others (echolalia) or repeating one's own words (palilalia). The involuntary use of inappropriate or obscene language (coprolalia) is possible, but uncommon, among people with Tourette syndrome. In addition to frequent tics, people with Tourette syndrome are at risk for associated problems including attention deficit hyperactivity disorder (ADHD), obsessive-compulsive disorder (OCD), anxiety, depression, and problems with sleep.\n\n"
        "What causes Prader-Willi syndrome ?\n"
        "Prader-Willi syndrome (PWS) is caused by the loss of active genes in a specific region of chromosome 15. People normally inherit one copy of chromosome 15 from each parent. Some genes on chromosome 15 are only active (or 'expressed') on the copy that is inherited from a person's father (the paternal copy). When genes are only active if inherited from a specific parent, it is called genomic imprinting. About 70% of cases of PWS occur when a person is missing specific genes on the long arm of the paternal copy of chromosome 15. This is called a deletion. While there are copies of these same genes on the maternal copy of chromosome 15, the maternal copies of these genes are not expressed. In about 25% of cases, PWS is due to a person inheriting only 2 maternal copies of chromosome 15, instead of one copy from each parent. This is called maternal uniparental disomy. Rarely (in about 2% of cases), PWS is caused by a rearrangement of chromosome material called a translocation, or by a change (mutation) or other defect that abnormally inactivates genes on the paternal chromosome 15. Each of these genetic changes result in a loss of gene function on part of chromosome 15, likely causing the characteristic features of PWS. \n\n"
        "How many people are affected by nonsyndromic paraganglioma ?\n"
        "It is estimated that the prevalence of pheochromocytoma is 1 in 500,000 people, and the prevalence of other paragangliomas is 1 in 1 million people. These statistics include syndromic and nonsyndromic paraganglioma and pheochromocytoma. \n\n"
        "What to do for Causes of Diabetes ?\n"
        "Diabetes is a complex group of diseases with a variety of causes. Scientists believe genes and environmental factors interact to cause diabetes in most cases. - People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes develops when the body doesnt make enough insulin or is not able to use insulin effectively, or both. - Insulin is a hormone made by beta cells in the pancreas. Insulin helps cells throughout the body absorb and use glucose for energy. If the body does not produce enough insulin or cannot use insulin effectively, glucose builds up in the blood instead of being absorbed by cells in the body, and the body is starved of energy. - Prediabetes is a condition in which blood glucose levels or A1C levels are higher than normal but not high enough to be diagnosed as diabetes. People with prediabetes can substantially reduce their risk of developing diabetes by losing weight and increasing physical activity. - The two main types of diabetes are type 1 diabetes and type 2 diabetes. Gestational diabetes is a third form of diabetes that develops only during pregnancy. - Type 1 diabetes is caused by a lack of insulin due to the destruction of insulin-producing beta cells. In type 1 diabetesan autoimmune diseasethe bodys immune system attacks and destroys the beta cells. - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 diabetes develops when the body can no longer produce enough insulin to compensate for the impaired ability to use insulin. - Scientists believe gestational diabetes is caused by the hormonal changes and metabolic demands of pregnancy together with genetic and environmental factors. Risk factors for gestational diabetes include being overweight and having a family history of diabetes. - Monogenic forms of diabetes are relatively uncommon and are caused by mutations in single genes that limit insulin production, quality, or action in the body. - Other types of diabetes are caused by diseases and injuries that damage the pancreas; certain chemical toxins and medications; infections; and other conditions.\n\n"
    )
    return examples

@app.post("/predict/")
async def predict(input_text: str):
    # Chain-of-Thought Prompting
    cot_text = cot_prompt(input_text)
    
    # Adding Few-Shot examples
    prompt = few_shot_prompt() + cot_text

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True)
    
    # Generate prediction with proper settings for T5
    outputs = model.generate(
        **inputs, 
        max_length=200,  # Maximum length for the response
        num_beams=5,  # Beam search for more coherent answers
        early_stopping=True,
        temperature=0.9,  # Controls randomness (lower is more focused)
        top_k=50,  # Use top-k sampling with k=50
        top_p=0.95,  # Use nucleus sampling with p=0.95
        repetition_penalty=1.2  # Penalize repetition
    )
    
    # Decode the model output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return {"response": response}

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
