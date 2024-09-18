import gradio as gr
import requests

# Function to process user input and generate a response
def generate_response(user_input, conversation_history):
    api_url = "http://34.172.3.202:8000/predict/"
    payload = {"input_text": user_input}

    try:
        # Make the POST request to the backend API
        response = requests.post(api_url, params=payload)
        
        # Extract the generated response from the JSON output
        if response.status_code == 200:
            assistant_reply = response.json().get("response", "Sorry, no response.")
        else:
            assistant_reply = "Error: Unable to process the request."
    
    except Exception as e:
        assistant_reply = f"Error: {str(e)}"

    conversation_history.append((user_input, assistant_reply))
    return "", conversation_history

# Define the Gradio Interface with styling
with gr.Blocks(css="""
    body {
        background: url('https://th.bing.com/th/id/R.d847a18d1fcb4cf637bfb330854b5770?rik=Tr74YewtX6zt7A&riu=http%3a%2f%2fwallpapercave.com%2fwp%2fvSb3JKZ.jpg&ehk=a2jZzaoaunt28ouTcCnwBIQslzDrb%2btZB0cOPVlB36k%3d&risl=&pid=ImgRaw&r=0') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Roboto', sans-serif;
    }
    .chat-window {
        background: linear-gradient(to right, #007bff, #f0f8ff); /* Blue to white gradient */
        background-image: url('https://th.bing.com/th/id/R.d847a18d1fcb4cf637bfb330854b5770?rik=Tr74YewtX6zt7A&riu=http%3a%2f%2fwallpapercave.com%2fwp%2fvSb3JKZ.jpg&ehk=a2jZzaoaunt28ouTcCnwBIQslzDrb%2btZB0cOPVlB36k%3d&risl=&pid=ImgRaw&r=0'); /* Replace with your image path */
        background-size: cover;
        background-position: center;
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(5px);
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    .submit-btn {
        background: linear-gradient(to right, #007bff, #004cff);
        color: white;
        border-radius: 5px;
    }
    .input-box {
        border: 1px solid #00796B;
        border-radius: 5px;
        font-size: 18px; /* Increased font size for input box */
    }
    h1 {
        color: #00796B;
        font-size: 24px; /* Increased font size for h1 */
    }
    .background-image {
        /* Removed as the stock image is now the background */
    }
""") as interface:

    gr.Markdown("""
        <div style='text-align: center;'>
            <h1>MedAssistAI</h1>
            <p style='font-size: 18px;'>Your Virtual Medical Assistant</p>
        </div>
    """)

    # State to keep track of the conversation
    conversation_history = gr.State([])

    # Main chat interface
    with gr.Column(scale=6, elem_classes="chat-window"):
        chat_display = gr.Chatbot(label="MediAssistAI", height=400)

        # Input section below the chat window
        with gr.Row():
            user_message = gr.Textbox(show_label=False, placeholder="Enter your message here...", elem_classes="input-box")
        with gr.Row():
            send_button = gr.Button("Send", elem_classes="submit-btn")

    # Trigger the response generation when the send button is clicked
    send_button.click(generate_response, [user_message, conversation_history], [user_message, chat_display])

# Launch the interface
interface.launch(share=True)
