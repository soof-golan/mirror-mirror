.PHONY: pipeline audio camera mic show_fmts loopback state_sync generate deploy help
.PHONY: screen_pipeline screen_audio screen_camera screen_vignette screen_ui
.PHONY: vignette ui start stop remote_start remote_stop _no_generate

REMOTE := beast-ubuntu
PROJECT_DIR := "~/dev/soof-golan/on-the-wall"

help:
	@echo "Available targets:"
	@echo "  remote_start  - Start all processes on remote in detached screens"
	@echo "  remote_stop   - Stop all processes on remote"
	@echo "  deploy        - Deploy to remote server"
	@echo ""
	@echo "Individual processes:"
	@echo "  pipeline      - Run the diffusion pipeline (publishes to IPC)"
	@echo "  vignette      - Run the vignette process (audio -> vignette intensity)"
	@echo "  ui            - Run the UI process"
	@echo "  audio         - Run the audio input module (Gemma 3n)"
	@echo "  camera        - Run the camera input module"
	@echo "  mic           - Run the microphone input module"
	@echo "  state_sync    - Run the state synchronization module"
	@echo ""
	@echo "Utilities:"
	@echo "  show_fmts     - Show supported video formats for /dev/video0"
	@echo "  loopback      - Run loopback debug mode"
	@echo "  generate      - Generate code from protobuf definitions"

deploy:
	git push && ssh $(REMOTE) "cd $(PROJECT_DIR) && git pull"

start:
	screen -dmS camera make camera
	screen -dmS state_sync make state_sync
	screen -dmS audio make audio
	screen -dmS pipeline make pipeline
	screen -dmS vignette make vignette
	screen -dmS ui make ui
	@echo "All processes started. Use 'make stop' to stop all."

stop:
	@screen -ls | grep -E "\.(camera|state_sync|audio|pipeline|vignette|ui)\s" | awk '{print $$1}' | xargs -I {} screen -S {} -X quit 2>/dev/null || true
	@screen -wipe 2>/dev/null || true
	@pkill -f "mirror\.(pipeline|vignette|audio|camera|state_sync|ui)" 2>/dev/null || true
	@pkill -f "uv run -m mirror\." 2>/dev/null || true
	@echo "All processes stopped."

remote_start:
	ssh $(REMOTE) "cd $(PROJECT_DIR) && make start"
remote_stop:
	ssh $(REMOTE) "cd $(PROJECT_DIR) && make stop"

# Screen sessions for remote (interactive)
screen_pipeline:
	ssh beast-ubuntu -t "cd $(PROJECT_DIR) && screen -D -R pipeline make pipeline"

screen_audio:
	ssh beast-ubuntu -t "cd $(PROJECT_DIR) && screen -D -R audio make audio"

screen_camera:
	ssh beast-ubuntu -t "cd $(PROJECT_DIR) && screen -D -R camera make camera"

screen_state_sync:
	ssh beast-ubuntu -t "cd $(PROJECT_DIR) && screen -D -R state_sync make state_sync"

screen_vignette:
	ssh beast-ubuntu -t "cd $(PROJECT_DIR) && screen -D -R vignette make vignette"

screen_ui:
	ssh beast-ubuntu -t "cd $(PROJECT_DIR) && screen -D -R ui make ui"

# Local processes
pipeline:
	uv run -m mirror.pipeline

vignette:
	uv run -m mirror.vignette

ui:
	DISPLAY=:1 uv run python ui/main.py

audio:
	uv run -m mirror.audio --mic-lookup tascam

camera:
	uv run -m mirror.native_camera

mic:
	uv run -m mirror.native_mic --lookup tascam

show_fmts:
	v4l2-ctl --device=/dev/video0 --list-formats-ext

loopback:
	uv run -m mirror.native_ui --debug

state_sync:
	uv run -m mirror.state_sync

generate: _no_generate mirror/generated/

_no_generate:
	rm -rf ./mirror/generated/

mirror/generated/:
	uv run python-grpc-tools-protoc -I . --python_betterproto2_out=mirror proto/*.proto
