from sanic import Sanic
from sanic.response import html
import asyncio
import pathlib
from live_video_feed import LiveLoopInference

slideshow_root_path = pathlib.Path(__file__).parent.joinpath("slideshow")

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)


@app.route("/")
async def index(request):
    return html(open(slideshow_root_path.joinpath("slideshow.html"), "r").read())

valid_commands = ["right", "left", "rotate_right", "rotate_left", "zoom_in", "zoom_out", "up", "down", "flip", "spin", "point"]

@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")

    # # ======================== add calls to your model here ======================
    # # uncomment for event emitting demo: the following loop will alternate
    # # emitting events and pausing
    while True:
        await asyncio.sleep(1)
        with open("command.txt", "r+") as file:
            content = file.read().strip()
            print("read content>" + content + "<")
            if content in valid_commands:
                print("emitting" + content)
                await ws.send(content)
                file.write("noop")
            else:
                print("no command in file")

if __name__ == "__main__":
    LiveLoopInference(inference_mode=0)
    app.run(host="127.0.0.1", debug=True)
