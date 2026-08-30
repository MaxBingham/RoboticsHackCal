import time
import datetime
import os
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from config import ELEVENLABS_API_KEY, AGENT_ID

OUTPUT_DIR = "grabbed_nuts_log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def grab_nuts(params):
    count = int(params.get("count"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{count}_nuts_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w") as f:
        f.write(f"Number of nuts: {count}\nTimestamp: {timestamp}\n")

    print("\n" + "="*50)
    print(f"  🤖 NUTS GRABBED")
    print(f"  Count:     {count}")
    print(f"  Timestamp: {timestamp}")
    print(f"  File:      {filepath}")
    print("="*50 + "\n")

    return f"{count} nuts grabbed"

client_tools = ClientTools()
client_tools.register("grab_nuts", grab_nuts, is_async=False)

elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

conversation = Conversation(
    client=elevenlabs,
    agent_id=AGENT_ID,
    requires_auth=True,
    audio_interface=DefaultAudioInterface(),
    client_tools=client_tools,
)

print("Starting conversation... say e.g. 'grab 6 nuts'. Press Ctrl+C to stop.")
conversation.start_session()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nEnding conversation...")
    conversation.end_session()
    conversation.wait_for_session_end()