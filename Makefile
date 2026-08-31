.PHONY: test install

test:
	python3 -m unittest discover -s tests -v

install:
	mkdir -p ~/.hermes/plugins/hermes-classifier-mode
	cp __init__.py plugin.yaml README.md LICENSE ~/.hermes/plugins/hermes-classifier-mode/
	@echo "Installed. Enable with: hermes plugins enable hermes-classifier-mode"
