import {motion} from 'framer-motion'
import { useEffect, useMemo, useState, useRef } from "react"
import { io, Socket } from 'socket.io-client';
import on_mic from '../assets/on_mic.svg'
import off_mic from '../assets/off_mic.svg'
function Voice() {
  const [scale, setScale] = useState(1);
  const [isSpeaking,setIsSpeaking] = useState(false)
  const [speaker, setSpeaker] = useState<string | null>("loading...");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const socket = useMemo<Socket>(()=> io('http://localhost:3000'), [])

  useEffect(()=>{
    socket.on("recognised_speaker", (data) => {
      setSpeaker(data)
    })
  },[])

  const startRecording = async ()=>{
    try{
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsSpeaking(true);
      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          socket.emit('audio', event.data);
        }
      }
    }
    catch(err){
      console.error("Error accessing microphone", err);
      return;
    }
  }

  const stopRecording = ()=>{
    if(mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsSpeaking(false);
      
    }

  }

  useEffect(() => {
    let interval = 0;
        interval = setInterval(() => {
      setScale(1+ Math.random());
    }, 100);

    return () => clearInterval(interval);
  }, []);
  return (
    <motion.div initial={{ opacity: 0, y:500, scaleX:0}} animate={{opacity:1,y:0,scaleX:1}} exit={{ opacity: 0, y: 500,scaleX:0}} transition={{ duration: 0.8 }} className="fixed inset-0 top-15 flex w-screen h-screen items-center justify-center bg-[rgba(0,0,0,0.5)] text-white rounded-t-3xl">
      <div className="relative -top-30 w-40 h-40 flex justify-center items-center">
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            // style = {{borderRadius: i}}
            className={`absolute border-blue-500 rounded-full opacity-60 transition-transform duration-500 ease-in-out`}
            style={{
              width: `${60 + i * 20}px`,
              height: `${60 + i * 20}px`,
              transform: `scale(${isSpeaking?scale:1})`,
              borderWidth: `${i+1}px`
            }}
          ></div>
        ))}
      <div className='text-sm'>
        {speaker? speaker:""}
      </div>
      </div>
      <div className='fixed cursor-pointer bg-[rgba(5,5,255,0.3)] bottom-10 text-3xl text-center font-sans w-[100px] h-[100px] flex justify-center items-center rounded-[100%] text-white px-4 py-2'>
        <div onClick={isSpeaking?stopRecording:startRecording} className='w-full flex my-auto'>
          <img src={!isSpeaking?on_mic:off_mic} alt="" />
          </div>
      </div>
    </motion.div>
  )
}

export default Voice