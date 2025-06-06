// import { Link } from "react-router-dom"

// TODO: Add Link Tags in Every Part and add proper Links
function Navbar() {
  return (
    <div className='w-full text-white flex text-2xl justify-end space-x-3 mt-4 h-fit'>
      <div className="w-1/2 flex justify-around font-mono h-fit">
        <div onClick={()=>window.location.href='https://github.com/bhardwajdevesh092005/prml_speaker_recog_spring_2025'} className="relative cursor-pointer after:content-[''] after:absolute after:left-0 after:bottom-0 after:w-0 after:h-[2px] after:bg-white after:transition-all after:duration-300 hover:after:w-full">Github</div>
        <div onClick={()=>window.location.href='https://bhardwajdevesh092005.github.io/PRML_Project_Page/index.html'} className="relative cursor-pointer after:content-[''] after:absolute after:left-0 after:bottom-0 after:w-0 after:h-[2px] after:bg-white after:transition-all after:duration-300 hover:after:w-full">Project Page</div>
        <div className="relative cursor-pointer after:content-[''] after:absolute after:left-0 after:bottom-0 after:w-0 after:h-[2px] after:bg-white after:transition-all after:duration-300 hover:after:w-full">Report</div>
        <div className="relative cursor-pointer after:content-[''] after:absolute after:left-0 after:bottom-0 after:w-0 after:h-[2px] after:bg-white after:transition-all after:duration-300 hover:after:w-full">Pipeline</div>
      </div>
    </div>
  )
}

export default Navbar