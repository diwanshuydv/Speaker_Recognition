import {motion} from 'framer-motion'
function Voice() {
  return (
    <motion.div initial={{ opacity: 0, y:500, scaleX:0}} animate={{opacity:1,y:0,scaleX:1}} exit={{ opacity: 0, y: -100,scale:1.2}} transition={{ duration: 0.8 }} className="fixed inset-0 top-15 flex w-screen h-screen items-center justify-center bg-[rgba(0,0,0,0.5)] text-white rounded-t-3xl">
      Main
    </motion.div>
  )
}

export default Voice