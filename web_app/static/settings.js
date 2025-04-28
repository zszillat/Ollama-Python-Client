let settings = window.settingsData;
let editingPresetIndex = null;

const ollamaOptionsFields = [
    "temperature", "top_p", "top_k", "repeat_penalty", "stop", "images"
];

const editPresetModal = document.getElementById('editPresetModal');

document.addEventListener("DOMContentLoaded", function() {
    const themeSelect = document.getElementById('theme');
    const presetSelect = document.getElementById('presets');
    const baseUrlInput = document.getElementById('base-url');

    settings.theme.themes.forEach(t => {
        const option = document.createElement('option');
        option.value = t;
        option.text = t;
        if (t === settings.theme.selected) option.selected = true;
        themeSelect.appendChild(option);
    });

    settings.manageModels.modelPresets.forEach(t => {
      const option = document.createElement('option');
      option.value = t.name;
      option.text = t.name;
      if (t.name === settings.presetDefault) {
        //console.log(t)
        option.selected = true;
      }
      presetSelect.appendChild(option);
    });

    baseUrlInput.value = settings.base_url;

    refreshModelPresets();

    themeSelect.addEventListener('change', () => {
        settings.theme.selected = themeSelect.value;
        saveSettings();
    });

    presetSelect.addEventListener('change', () => {
      settings.presetDefault = presetSelect.value;
      saveSettings();
    });

    baseUrlInput.addEventListener('blur', () => {
        settings.base_url = baseUrlInput.value;
        saveSettings();
    });

    const closeEditPresetBtn = document.getElementById('closeEditPreset');

    if (closeEditPresetBtn) {
        closeEditPresetBtn.onclick = () => editPresetModal.style.display = "none";
    }

    const savePresetChangesBtn = document.getElementById('save-preset-changes');

    if (savePresetChangesBtn) {
        savePresetChangesBtn.addEventListener('click', function() {

            if (editingPresetIndex !== null) {
                const newPreset = {
                    model: document.getElementById('preset-model')?.value || "",
                    name: document.getElementById('preset-name')?.value || "",
                    options: {}
                };

                ollamaOptionsFields.forEach(field => {
                    if (field === "stop" || field === "images") {
                        const listItems = document.getElementById(`${field}-list`)?.querySelectorAll('span') || [];
                        const values = Array.from(listItems).map(el => el.textContent);
                        if (values.length > 0) newPreset.options[field] = values;
                    } else {
                        const elem = document.getElementById(`preset-${field}`);
                        if (elem) {
                            if (elem.type === "checkbox") {
                                if (elem.checked) newPreset.options[field] = true;
                            } else {
                                const val = elem.value.trim();
                                if (val !== "") {
                                    if (!isNaN(val)) {
                                        newPreset.options[field] = Number(val);
                                    } else {
                                        newPreset.options[field] = val;
                                    }
                                }
                            }
                        }
                    }
                });

                settings.manageModels.modelPresets[editingPresetIndex] = newPreset;
                refreshModelPresets();
                saveSettings();
                editPresetModal.style.display = "none";
            }
        });
    }
});

function refreshModelPresets() {
    const presetsList = document.getElementById('model-presets-list');
    if (!presetsList) return;
    presetsList.innerHTML = "";
    settings.manageModels.modelPresets.forEach((preset, index) => {
        const li = document.createElement('li');
        li.textContent = preset.name;

        const editButton = document.createElement('button');
        editButton.type = "button"; // <- ADD THIS
        editButton.textContent = "✏️ Edit";
        editButton.style.marginLeft = "10px";
        editButton.onclick = () => openEditPreset(index);

        const deleteButton = document.createElement('button');
        deleteButton.type = "button"; // <- ADD THIS
        deleteButton.textContent = "❌ Delete";
        deleteButton.style.marginLeft = "5px";
        deleteButton.onclick = () => {
            settings.manageModels.modelPresets.splice(index, 1);
            refreshModelPresets();
            saveSettings();
        };


        li.appendChild(editButton);
        li.appendChild(deleteButton);
        presetsList.appendChild(li);
    });
}

function openEditPreset(index) {
    editingPresetIndex = index;
    const preset = settings.manageModels.modelPresets[index];

    const modelInput = document.getElementById('preset-model');
    const nameInput = document.getElementById('preset-name');
    const advancedFields = document.getElementById('advanced-fields');
    const toggleAdvancedBtn = document.getElementById('toggle-advanced-fields');

    if (modelInput) modelInput.value = preset.model || "";
    if (nameInput) nameInput.value = preset.name || "";

    if (advancedFields) {
        advancedFields.style.display = 'none';
        advancedFields.innerHTML = '';
    }
    if (toggleAdvancedBtn) {
        toggleAdvancedBtn.textContent = 'Show Advanced';
        toggleAdvancedBtn.onclick = () => {
            if (advancedFields.style.display === 'none') {
                createAdvancedFields();
                fillAdvancedFields(preset.options);
                advancedFields.style.display = 'block';
                toggleAdvancedBtn.textContent = 'Hide Advanced';
            } else {
                advancedFields.style.display = 'none';
                toggleAdvancedBtn.textContent = 'Show Advanced';
            }
        };
    }

    editPresetModal.style.display = "block";
}

function fillAdvancedFields(options) {
    if (!options) return;
    ollamaOptionsFields.forEach(field => {
        const elem = document.getElementById(`preset-${field}`);
        if (!elem) return;

        const val = options[field];
        if (val !== undefined) {
            if (field === "stop" || field === "images") {
                const listContainer = document.getElementById(`${field}-list`);
                if (Array.isArray(val) && listContainer) {
                    listContainer.innerHTML = '';
                    val.forEach(item => {
                        const span = document.createElement('span');
                        span.textContent = item;
                        listContainer.appendChild(span);
                    });
                }
            } else {
                elem.value = val;
            }
        }
    });
}

function createAdvancedFields() {
    const container = document.getElementById('advanced-fields');
    if (!container) return;

    container.innerHTML = "";

    ollamaOptionsFields.forEach(field => {
        const div = document.createElement('div');
        div.className = "form-group";

        const label = document.createElement('label');
        label.textContent = field.charAt(0).toUpperCase() + field.slice(1) + ":";
        label.setAttribute('for', `preset-${field}`);

        let input;
        if (field === "stop" || field === "images") {
            input = document.createElement('div');
            input.id = `${field}-list`;
            input.className = "list-container";
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.id = `preset-${field}`;
        }

        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    });
}

async function saveSettings() {
    await fetch('/save_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    });
}