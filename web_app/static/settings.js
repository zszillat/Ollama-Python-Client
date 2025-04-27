let settings = window.settingsData;
let editingPresetIndex = null;

// Modals
const installedModal = document.getElementById('installedModelsModal');
const presetsModal = document.getElementById('modelPresetsModal');
const editPresetModal = document.getElementById('editPresetModal');

document.addEventListener("DOMContentLoaded", function() {
    const themeSelect = document.getElementById('theme');
    const baseUrlInput = document.getElementById('base-url');

    settings.theme.themes.forEach(t => {
        const option = document.createElement('option');
        option.value = t;
        option.text = t;
        if (t === settings.theme.selected) option.selected = true;
        themeSelect.appendChild(option);
    });

    baseUrlInput.value = settings.base_url;

    refreshInstalledModels();
    refreshModelPresets();

    themeSelect.addEventListener('change', () => {
        settings.theme.selected = themeSelect.value;
        saveSettings();
    });

    baseUrlInput.addEventListener('blur', () => {
        settings.base_url = baseUrlInput.value;
        saveSettings();
    });

    document.getElementById('manage-installed-models').onclick = () => installedModal.style.display = "block";
    document.getElementById('manage-model-presets').onclick = () => presetsModal.style.display = "block";
    document.getElementById('closeInstalledModels').onclick = () => installedModal.style.display = "none";
    document.getElementById('closeModelPresets').onclick = () => presetsModal.style.display = "none";
    document.getElementById('closeEditPreset').onclick = () => editPresetModal.style.display = "none";

    document.getElementById('add-installed-model').addEventListener('click', function() {
        const newModelInput = document.getElementById('new-installed-model');
        const newModel = newModelInput.value.trim();
        if (newModel) {
            settings.manageModels.modelInstalled.push(newModel);
            newModelInput.value = "";
            refreshInstalledModels();
            saveSettings();
        }
    });

    document.getElementById('save-preset-changes').addEventListener('click', function() {
        if (editingPresetIndex !== null) {
            try {
                const newPreset = JSON.parse(document.getElementById('edit-preset-json').value);
                settings.manageModels.modelPresets[editingPresetIndex] = newPreset;
                refreshModelPresets();
                saveSettings();
                editPresetModal.style.display = "none";
            } catch (e) {
                alert("Invalid JSON format.");
            }
        }
    });
});

function refreshInstalledModels() {
    const installedList = document.getElementById('installed-models-list');
    installedList.innerHTML = "";
    settings.manageModels.modelInstalled.forEach((model, index) => {
        const li = document.createElement('li');
        li.textContent = model;

        const deleteButton = document.createElement('button');
        deleteButton.textContent = "❌";
        deleteButton.style.marginLeft = "10px";
        deleteButton.onclick = () => {
            settings.manageModels.modelInstalled.splice(index, 1);
            refreshInstalledModels();
            saveSettings();
        };

        li.appendChild(deleteButton);
        installedList.appendChild(li);
    });
}

function refreshModelPresets() {
    const presetsList = document.getElementById('model-presets-list');
    presetsList.innerHTML = "";
    settings.manageModels.modelPresets.forEach((preset, index) => {
        const li = document.createElement('li');
        li.textContent = preset.name;

        const editButton = document.createElement('button');
        editButton.textContent = "✏️ Edit";
        editButton.style.marginLeft = "10px";
        editButton.onclick = () => openEditPreset(index);

        const deleteButton = document.createElement('button');
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
    document.getElementById('edit-preset-json').value = JSON.stringify(preset, null, 4);
    editPresetModal.style.display = "block";
}

async function saveSettings() {
    await fetch('/save_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    });
}
