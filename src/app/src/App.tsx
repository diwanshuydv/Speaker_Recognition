import './App.css'
import Home from './Components/Home'
import { useState } from 'react'
import Navbar from './Components/Navbar'
import { useSwipeable } from 'react-swipeable'
import { AnimatePresence } from 'framer-motion'
import Voice from './Components/Voice'
function App() {
  // const [animating,setAnimating] = useState(false)
  const [isHome,setIsHome] = useState(true)
  const [homeStyles,setHomeStyles] = useState({ opacity: 1, y: 0, scale:1 })
  // const [bgClass,setBgClass] = useState()
  const handlers = useSwipeable({
      onSwipedUp: ()=>{
        window.location.href = '/home'
      },
      // preventDefaultTouchmoveEvent: true,
      trackTouch: true,
    })
  const handleWheel = (e:React.WheelEvent<HTMLDivElement>)=>{
    if(e.deltaY>50){
      setIsHome(false);
    }else if(e.deltaY<-50){
      setHomeStyles({ opacity: 0, y: -100,scale:1.2})
      setIsHome(true);
    }
  }
  return (
    <div {...handlers} onWheel={handleWheel} className={`bg-[url(/bg.jpg)] bg-cover bg-center  w-screen overflow-hidden h-screen`}>
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
        {[...Array(30)].map((_, i) => (
          <span
            key={i}
            className="absolute text-white"
            style={{
              left: `${Math.random() * 100}vw`,
              top: `-${Math.random() * 50}vh`,
              animation: `fall ${2 + Math.random()}s ease-in-out infinite`,
              fontSize: `${Math.random() * 10 + 10}px`,
            }}
          >
            ❄️
          </span>
        ))}
      </div>

      {/* CSS Animation */}
      <style>
        {`
          @keyframes fall {
            0% {
              transform: translateY(0) rotate(0deg);
              opacity: 1;
            }
            100% {
              transform: translateY(50vh) rotate(360deg);
              opacity: 0;
            }
          }
        `}
      </style>
      <Navbar/>
      <AnimatePresence mode="wait">
      {isHome&&(<Home styles={homeStyles}/>)}
      </AnimatePresence>
      <AnimatePresence>
        {!isHome&&(<Voice/>)}
      </AnimatePresence>
    </div>
  )
}

export default App
