// Loopback-only preview, limited to the four public review assets.
const http=require("node:http");
const fs=require("node:fs");
const path=require("node:path");
const root=path.resolve(__dirname,"../../../videns-site");
const files={"/review.html":["review.html","text/html; charset=utf-8"],
  "/review-playback.js":["review-playback.js","application/javascript"],
  "/review-faker-data.js":["review-faker-data.js","application/javascript"],
  "/review-faker-evidence.json":["review-faker-evidence.json","application/json"]};
const server=http.createServer((request,response)=>{
  const asset=files[new URL(request.url,"http://127.0.0.1").pathname];
  if(request.method!=="GET"||!asset){response.writeHead(404);response.end();return;}
  fs.readFile(path.join(root,asset[0]),(error,bytes)=>{
    if(error){response.writeHead(500);response.end("Preview asset unavailable");return;}
    response.writeHead(200,{"Content-Type":asset[1],"Cache-Control":"no-store"});response.end(bytes);
  });
});
server.listen(8765,"127.0.0.1",()=>console.log("Review preview: http://127.0.0.1:8765/review.html"));
