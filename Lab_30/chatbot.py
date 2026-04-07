import sys

def simple_chatbot(user_input):
    """
    Logic function that determines the response based on fixed rules.
    """
    # Normalize input to lowercase to handle 'Hello', 'HELLO', or 'hello'
    user_input = user_input.lower().strip()

    # Task 1 & 2: Rule-Based Logic via If/Else
    if user_input == "hello" or user_input == "hi":
        return "Hi there! How can I help you today?"
    
    elif user_input == "what is your name?":
        return "I am a simple chatbot created for Lab 30."
    
    elif "how are you" in user_input:
        return "I'm doing great, thank you! Ready to help you with your lab."
    
    elif "weather" in user_input:
        return "I don't have a thermometer, but your CPU seems to be running just fine!"
    
    elif user_input == "what is nlp?":
        return "NLP stands for Natural Language Processing, which you studied in Lab 27!"
    
    elif user_input == "bye":
        return "Goodbye! Have a great day!"
    
    else:
        # The 'Catch-all' response for unknown inputs
        return "I'm sorry, I didn't understand that. Could you try asking something else?"

def run_chatbot():
    """
    Main loop to handle the user interface in the terminal.
    """
    print("--- Simple Rule-Based Chatbot ---")
    print("Welcome! (Type 'bye' to exit)")
    
    while True:
        try:
            # Task 3: Get user input
            user_input = input("You: ")
            
            # Get response from the logic function
            response = simple_chatbot(user_input)
            
            # Print the response
            print("Bot: " + response)
            
            # Exit condition
            if user_input.lower().strip() == "bye":
                break
        
        except EOFError:
            # Handles Ctrl+D gracefully
            break
        except KeyboardInterrupt:
            # Handles Ctrl+C gracefully
            print("\nBot: Goodbye!")
            break

if __name__ == "__main__":
    run_chatbot()
