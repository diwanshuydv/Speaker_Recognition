import { RefAttributes, useState } from "react";
// import { useNavigate } from "react-router-dom"
import { HTMLMotionProps, motion } from "framer-motion";
interface HomeProps {
  styles: IntrinsicAttributes & Omit<HTMLMotionProps<"div">, "ref"> & RefAttributes<HTMLDivElement>;
}

function Home({ styles }: HomeProps) {
  const [animating, setAnimating] = useState(true);
  // const navigate = useNavigate()
  return (
    <motion.div
      initial={styles}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -100, scale: 0 }}
      transition={{ duration: 0.8 }}
      onAnimationEnd={() => setAnimating(false)}
      className="text-5xl text-center text-white font-sans mt-20 "
    >
      <h1 className="animate-fade-in-once text-blue-900">
        Speaker Recognition
      </h1>
      <button
        className={`${!animating ? "animate-Bounce-once" : "hidden"} text-3xl`}
      >
        Continue
      </button>
    </motion.div>
  );
}

export default Home;
