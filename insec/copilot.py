#!/usr/bin/env python3
import dataclasses
import functools
import re
import subprocess
import json
import threading
from pathlib import Path
from time import sleep, time

from fire import Fire
from nodejs import node

# ----------- LSP ---------------------

def discard(_):
    pass

@dataclasses.dataclass
class CopilotServer:
    server_process: subprocess.Popen
    request_id = 0
    resolve_map: dict = dataclasses.field(default_factory=dict)
    reject_map: dict = dataclasses.field(default_factory=dict)
    view_id = 0
    panel_id = 0

    # Code for communicating with the Language Server through LSP

    def send_message(self, data):
        data.update({"jsonrpc": "2.0"})
        data_string = json.dumps(data)
        content_length = len(data_string.encode("utf-8"))
        rpc_string = f"Content-Length: {content_length}\r\n\r\n{data_string}".encode("utf8")
        # print(f"Sending:\n\n {rpc_string}")
        self.server_process.stdin.write(rpc_string)
        self.server_process.stdin.flush()

    def send_request(self, method, params, cb_success=print, cb_error=print):
        self.request_id += 1
        self.send_message({"id": self.request_id, "method": method, "params": params})
        self.resolve_map[self.request_id] = cb_success
        self.reject_map[self.request_id] = cb_error

    def send_notification(self, method, params):
        self.send_message({"method": method, "params": params})

    def handle_received_payload(self, payload):
        if "id" in payload:
            if "result" in payload:
                # print("Request {payload_id} received".format(payload_id=payload["id"]))
                self.resolve_map.get(payload["id"], discard)(payload["result"])
            elif "error" in payload:
                # print("Request {payload_id} rejected".format(payload_id=payload["id"]))
                self.reject_map.get(payload["id"], discard)(payload["error"])
                raise RuntimeError(payload["error"])
        elif "method" in payload:
            self.resolve_map.get(payload["method"], discard)(payload["params"])

    def close(self):
        self.server_process.kill()


# Spawning the server process
LANG_SERVER_PATH = (
    Path(__file__).parent.parent.joinpath("lang-server").joinpath("agent.js")
)


def spawn_server():
    """Spawn a Copilot server"""

    def listen_to_server(server: CopilotServer):
        server_process = server.server_process
        while True:
            if server_process.poll() is not None:
                print("Server died")
                break
            read_len = len("Content-Length: ".encode("utf-8"))
            output = server_process.stdout.read(read_len).decode("utf-8")
            if not output:
                print("Server died")
                break
            assert output == "Content-Length: ", f"Expect Content-Length but got {output}"
            content_length = []
            while True:
                output = server_process.stdout.read(1).decode("utf-8")
                if not re.match(r"[0-9]", output):
                    break
                content_length.append(output)
            server_process.stdout.read(3)
            try:
                content_length = int("".join(content_length))
            except:
                raise ValueError(f"Expected integer but got {''.join(content_length)}")
            output = server_process.stdout.read(content_length).decode("utf8")
            # print(f"Reading: {output}")
            if output:
                payload_string = output
                try:
                    payload = json.loads(payload_string)
                    # print(payload)
                    server.handle_received_payload(payload)
                except json.JSONDecodeError as e:
                    print(f"Unable to parse payload: {payload_string}", e)

    process = node.Popen(
        [LANG_SERVER_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    server = CopilotServer(process)

    # Start a thread to listen to server's stdout
    thread = threading.Thread(target=listen_to_server, args=[server], daemon=True)
    thread.start()
    return server


# ----------- CoPilot ---------------------


@dataclasses.dataclass(frozen=True)
class Position:
    line: int
    character: int


def request_completion(
    server: CopilotServer,
    cb,
    text: str,
    position: Position,
    languageId="python",
    filename="file:///home/fakeuser/my-project/test.py",
):
    server.view_id += 1

    server.send_notification(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": filename,
                "languageId": languageId,
                "text": text,
                "version": server.view_id,
            }
        },
    )

    def _upon_completions(payload):
        cb(payload["completions"])

    server.send_request(
        "getCompletions",
        {
            "doc": {
                "uri": filename,
                "languageId": languageId,
                "position": {"line": position.line, "character": position.character},
                "version": server.view_id,
            }
        },
        _upon_completions,
    )


def request_completion_cycle(
    server: CopilotServer,
    cb,
    position: Position,
    languageId="python",
    filename="file:///home/fakeuser/my-project/test.py",
):
    def _upon_completions(payload):
        cb(payload["completions"])

    server.send_request(
        "getCompletionsCycling",
        {
            "doc": {
                "uri": filename,
                "languageId": languageId,
                "position": {"line": position.line, "character": position.character},
                "version": server.view_id,
            }
        },
        _upon_completions,
    )

def request_panel_completion(
        server: CopilotServer,
        cb,
        text: str,
        position: Position,
        languageId="python",
        filename="file:///home/fakeuser/my-project/test.py",
):
    server.view_id += 1
    server.panel_id += 1

    server.send_notification(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": filename,
                "languageId": languageId,
                "text": text,
                "version": server.view_id,
            }
        },
    )

    solutions = []
    target_solutions = [0]

    def collect_solutions(payload):
        solutions.append(payload)

    server.resolve_map["PanelSolution"] = collect_solutions

    def return_solutions(payload):
        cb(solutions)

    server.resolve_map["PanelSolutionsDone"] = return_solutions

    def _upon_completions(payload):
        target_solutions[0] = payload["solutionCountTarget"]

    server.send_request(
        "getPanelCompletions",
        {
            "doc": {
                "uri": filename,
                "languageId": languageId,
                "position": {"line": position.line, "character": position.character},
                "version": server.view_id,
            },
            "panelId": str(server.panel_id),
        },
        _upon_completions,
    )
def request_all_methods(
        server: CopilotServer,
        cb,
):
    def _upon_completions(payload):
        cb(payload)

    server.send_request(
        "getPanelCompletions",
        {
        },
        _upon_completions,
    )

def login(server: CopilotServer, cb, sign_out=False):
    def _on_signin_initiated(payload):
        if payload["status"] == "AlreadySignedIn":
            if sign_out:
                print("Signing out...")
                server.send_request("signOut", {}, lambda _: login(server, cb, sign_out=False))
                return
            cb(payload)
            return
        user_code = payload.get("userCode", "")
        verification_uri = payload.get("verificationUri", "")
        print("Please open the following url to authenticate")
        print(verification_uri)
        print("Enter the following code to sign in:")
        print(user_code)
        input("Press Enter to continue after signing in...")
        server.send_request("signInConfirm", {"userCode": user_code}, cb)

    server.send_request("signInInitiate", {}, _on_signin_initiated)


def initialize(server: CopilotServer, cb):
    def _on_initialized(r):
        server.send_notification("initialize", {})
        cb(r)

    #  Send `initialize` request
    server.send_request(
        "initialize",
        {
            "capabilities": {"workspace": {"workspaceFolders": True}},
        },
        _on_initialized,
    )


def await_fn(server: CopilotServer, fn, timeout: float = 100):
    """
    timeout: how long to wait until an error is raised in seconds
    """
    res = []

    def callback(server_res):
        res.append(server_res)

    fn(server, callback)

    start = time()
    while not res:
        if timeout is not None and time() - start > timeout:
            raise ValueError("Timed out waiting for result")
        sleep(0.1)
    return res[0]


def completions(
    server: CopilotServer,
    text='# hello world!\ndef query(cursor, id):\n    #  MUT LAYs STORAGE SYMüüü\n    cursor.execute("SELECT * FROM users WHERE id = \n    return cursor.fetchall()',
    position=Position(3, 50),
    language="python",
    sign_out: bool = False,
):
    cb = lambda server, callback: initialize(
        server,
        lambda _: login(
            server,
            lambda _: request_panel_completion(
                server,
                callback,
                text=text,
                position=position,
                languageId=language,
            ),
            sign_out=sign_out,
        ),
    )
    res1 = await_fn(server, cb)
    # res2 = await_fn(
    #     server,
    #     functools.partial(
    #         request_completion, position=position, languageId=language, text=text
    #     ),
    # )
    return len(res1), res1


if __name__ == "__main__":
    res1 = completions(spawn_server())
    res2 = completions(spawn_server())
    print(res1)
    print(res2)
    print(set(x["displayText"] for x in res1[1]) - set(x["displayText"] for x in res2[1]))
