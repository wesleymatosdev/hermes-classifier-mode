.PHONY: test install sync

test:
	python3 -m unittest discover -s tests -v

# The installed copy Hermes imports is the package dir. Repo root __init__.py
# and hermes_classifier_mode/__init__.py must be byte-identical at all times
# (see .hermes.md); 'sync' verifies it, 'install' copies to the plugin dir.
sync:
	cmp __init__.py hermes_classifier_mode/__init__.py

install:
	mkdir -p ~/.hermes/plugins/hermes-classifier-mode/hermes_classifier_mode
	cp __init__.py plugin.yaml README.md LICENSE ~/.hermes/plugins/hermes-classifier-mode/
	cp hermes_classifier_mode/__init__.py ~/.hermes/plugins/hermes-classifier-mode/hermes_classifier_mode/
	@echo "Installed. Enable with: hermes plugins enable hermes-classifier-mode"
