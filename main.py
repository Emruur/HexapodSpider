from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import time
import serial
import speech_recognition as sr

def send_rc100_button_bt410(ser, button_code):
    
    # port = '/dev/tty.usbserial-AB0MI2NT'
    # ser = serial.Serial(port, baudrate=57600, timeout=1)
    # time.sleep(2)

    # RC-100 Zigbee packet format: FF 55 Data_L ~Data_L 00
    # data_l = button_code
    # data_l_inv = (~data_l) & 0xFF  # one's complement
    # packet = bytearray([0xFF, 0x55, data_l, data_l_inv, 0x00])
    
    # print(f"Sending: {[hex(b) for b in packet]}")
    # ser.write(packet)
    # ser.flush()
    # time.sleep(0.1)
    
    def send_packet(code):
        data_l = code
        data_l_inv = (~data_l) & 0xFF
        packet = bytearray([0xFF, 0x55, data_l, data_l_inv, 0x00, 0xFF])
        print(f"Sending: {[hex(b) for b in packet]}")
        ser.write(packet)
        ser.flush()
        time.sleep(0.5)

    send_packet(button_code)
    time.sleep(0.1)
    # send_packet(0)# simulate releasing bottom
    
    # ser.close()

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Please speak...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio, language="en-US")
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Can not recognize speech")
        return None
    except sr.RequestError as e:
        print(f"Error: {e}")
        return None
    


def ask_llm(user_input):
    # prompt = f"""You are a speech command decoder. Given a sentence, you must convert it into a list of command codes, following this rule:

    #             - Move Forward = 1
    #             - Move Backward = 2
    #             - Turn Left = 4
    #             - Turn Right = 8
    #             - Ready = 16
    #             - Sit Down = 64
    #             - Others = 0

    #             Rules:
    #             - Repeat the number if the user says "two steps" or similar.
    #             - Output only a Python list. Do not include any explanation or text.

    #             Example:
    #             Input: "move forward"
    #             Output: [1]

    #             Input: "move forward two steps"
    #             Output: [1, 1]

    #             Input: "turn left and sit down"
    #             Output: [4, 64]

    #             Now convert this:
    #             {user_input}

    #             Output:

    # """
    
    # prompt = f"""<|system|>
    #             You are a speech command decoder. Output a Python list of command codes only. You can use synonyms to map codes.

    #             Command Map:
    #             - Move Forward = 1
    #             - Move Backward = 2
    #             - Turn Left = 4
    #             - Turn Right = 8
    #             - Ready = 16
    #             - Sit Down = 64
    #             - Unknown = 0

    #             Respond with only the list.
    #             For example:
    #             - If user inputs "Move forward two steps", return [1, 1].
    #             - If user inputs "Can you dancing?", return [1, 1, 2, 2, 4, 4, 8, 8]
    #             - If user inputs "Backward backward backward", return [2, 2, 2]
    #             - If user inputs "Rotate clockwise", return [8, 8, 8, 8]

    #             <|user|>
    #             {user_input}
    #             <|assistant|>
    #             """
    prompt = f"""<|system|>
    You are a speech command decoder. Output only a Python list of integers based on the following command codes:

    - Move Forward = 1  
    - Move Backward = 2  
    - Turn Left = 4  
    - Turn Right = 8  
    - Ready = 16
    - Attack = 32
    - Sit Down = 64  
    - Unknown = 0
    - Reinforcement Move Forward = 128
    - Reinforcement Move Backward = 130

    You may use synonyms. Only output a list of integers — no explanation or extra text.

    Examples:
    - Input: "Move forward two steps" Output: [1, 1]  
    - Input: "Can you dance?" Output: [1, 1, 2, 2, 4, 4, 8, 8]  
    - Input: "Backward backward backward" Output: [2, 2, 2]  
    - Input: "Rotate clockwise" Output: [8, 8, 8, 8]
    - Input: "Do RL move forward" Output: [128]
    - Input: "Do RL move forward two steps" Output: [128, 128]
    - Input: "Do Reinforcement learning move backward" Output: [130]

    <|user|>
    {user_input}
    <|assistant|>
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response)
    try:
        # start = response.find("{")
        # end = response.rfind("}") + 1
        if prompt in response:
            ans = response[len(prompt):]
        else:
            ans = response
        start = ans.find("[")
        end = ans.rfind("]")+1
        ans = ans[start:end]
        print(f"LLM answer:{ans}")
        return eval(ans)
    except:
        return {"action": "error", "steps": []}
    
def execute_llm_command(ser, command):
    result = ask_llm(command)
    print("Final decoded command list:", result)
    # if result["action"] == "error":
    #     print("decode llm fail")
    #     return
    for action in result:
        send_rc100_button_bt410(ser, action)
        time.sleep(3)
        
    # for step in result.get("steps", []):
    #     send_rc100_button_bt410(ser, step["code"])
    #     time.sleep(step.get("delay", 0.5))
        
if __name__ == "__main__":
    
    model_path = "./Qwen-Qwen1.5-32B-Chat"  


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # for decoder only model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto" 
    )
    
    # for encoder-decoder model
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("Model loaded!")
    
    ser = serial.Serial('/dev/tty.usbserial-AB0MI2NT', baudrate=57600, timeout=1)
    time.sleep(2)
    
    while True:
        command = recognize_speech()
        if command:
            # print(f"you said: {command}")
            execute_llm_command(ser, command)