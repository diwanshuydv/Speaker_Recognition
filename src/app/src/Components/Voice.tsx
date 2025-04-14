import { motion } from "framer-motion";
import { useEffect, useMemo, useState, useRef } from "react";
import { io, Socket } from "socket.io-client";
import on_mic from "../assets/on_mic.svg";
import off_mic from "../assets/off_mic.svg";

function Voice() {
  const [scale, setScale] = useState(1);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speaker, setSpeaker] = useState<string | null>("loading...");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const socket = useMemo<Socket>(() => io("http://127.0.0.1:3001"), []);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [file, setFile] = useState<File>();
  
  useEffect(() => {
    socket.on("recognised_speaker", (data: { data: string }) => {
      setSpeaker(data.data);
    });
  }, [socket]);

  const sendAudioData = async () => {
    if (audioChunksRef.current.length > 0) {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      audioChunksRef.current = []; // Clear after sending
      const buffer = await audioBlob.arrayBuffer();
      const audioBuffer = new Uint8Array(buffer);
      socket.emit("audio", audioBuffer);
      console.log("Audio data sent to server");
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsSpeaking(true);
      console.log("Microphone access granted!");
      const mimeType = "audio/webm"; // or try 'audio/ogg', 'audio/wav' depending on browser
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType }); 
      
      mediaRecorderRef.current.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
          sendAudioData(); 
        }
      };

      mediaRecorderRef.current.start(3000);
      mediaRecorderRef.current.onstop = () => {
        sendAudioData();
      };
      // Start sending audio every 3 seconds
      // intervalRef.current = setInterval(sendAudioData, 3000);
    } catch (err) {
      console.error("Error accessing microphone", err);
    }
  };

  const stopRecording = async () => {
    if (mediaRecorderRef.current) {
      setIsSpeaking(false);
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
    
    // Stop sending intervals
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Send remaining audio
    sendAudioData();
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setScale(0.5 + Math.random());
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 500, scaleX: 0 }}
      animate={{ opacity: 1, y: 0, scaleX: 1 }}
      exit={{ opacity: 0, y: 500, scaleX: 0 }}
      transition={{ duration: 0.8 }}
      className="fixed inset-0 top-15 flex flex-col w-screen h-screen items-center justify-center bg-[rgba(0,0,0,0.5)] text-white rounded-t-3xl"
    >
      <div className="relative -top-30 w-40 h-40 flex justify-center items-center">
      <div className="text-3xl text-blue-200"><b>{speaker ? speaker : ""}</b></div>
        {[...Array(7)].map((_, i) => (
          <div
            key={i}
            className="absolute border-blue-500 rounded-full opacity-60 transition-transform duration-500 ease-in-out"
            style={{
              width: `${90 + i * 25}px`,
              height: `${90 + i * 25}px`,
              transform: `scale(${isSpeaking ? scale : 1})`,
              borderWidth: `${i + 1}px`,
            }}
          ></div>
        ))}
      </div>
      <div
        className="fixed cursor-pointer bg-[rgba(70,70,253,0.53)] bottom-10 text-3xl text-center font-sans w-[100px] h-[100px] flex justify-center items-center rounded-[100%] text-white px-4 py-2"
        onClick={isSpeaking ? stopRecording : startRecording}
      >
        <img src={!isSpeaking ? on_mic : off_mic} alt="" />
      </div>
      <div className="flex flex-col space-y-2 items-center justify-center p-2 text-center">   
        {/* <div className="flex flex-col items-center justify-center p-2 text-center"> */}
        <input className="hidden" type="file" onChange={(e)=>setFile(e.target.files[0])} name="" id="audio_file" />
        <label htmlFor="audio_file" className="bg-black p-3 text-2xl rounded-full">Upload Audio</label>
        <div className="bg-black rounded-full p-3 text-xl">{file?.name}</div>

        <button className="bg-black p-3 text-2xl rounded-full" onClick={async ()=>{
          const formData = new FormData();
          formData.append("file", file);
          
          try {
            const response = await fetch("http://localhost:3001/api/upload", {
              method: "POST",
              body: formData,
            });
        
            const result = await response.json();
            console.log("Speaker Prediction:", result);
            setSpeaker(result.result);
          } catch (error) {
            console.error("Error uploading audio:", error);
          }
        }}>
          Submit
        </button>
        </div>
      {/* </div> */}
    </motion.div>
  );
}

export default Voice;
