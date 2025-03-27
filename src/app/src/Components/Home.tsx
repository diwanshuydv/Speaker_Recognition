import { useState } from "react"
// import { useNavigate } from "react-router-dom"
import {motion} from 'framer-motion'
function Home() {
  
    const [animating,setAnimating] = useState(true)
    // const navigate = useNavigate()
  return (
    
      <motion.div initial={{ opacity: 1, y: 0, scale:1}} exit={{ opacity: 0, y: -100,scale:1.2}} transition={{ duration: 0.8 }} onAnimationEnd={()=>setAnimating(false)} className="text-7xl text-center text-white font-sans mt-20 ">
          <h1 className="animate-fade-in-once text-blue-900">
              Welcome
          </h1>
          <button className={`${!animating?"animate-Bounce-once":"hidden"} text-5xl`}>
              Continue
          </button>
      </motion.div>
  )
}

export default Home