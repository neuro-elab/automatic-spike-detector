import os.path
from typing import List

import mne
import pyedflib
from mne.io import RawArray
from pyedflib import FILETYPE_EDFPLUS

from spidet.load.data_loading import DataLoader


def filter_leads_and_re_reference(
    file_path: str, channel_paths: List[str], leads: List[str]
):
    path = file_path[: file_path.rfind("/")]
    filename, ext = os.path.splitext(file_path[file_path.rfind("/") + 1 :])
    raw: RawArray = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    data_loader = DataLoader()

    channel_names = data_loader.extract_channel_names(channel_paths)

    raw = raw.pick(channel_names)

    raw = data_loader.generate_bipolar_references(raw, leads)

    dmin, dmax = -32768, 32767

    sfreq = raw.info["sfreq"]
    date = raw.info["meas_date"]

    # convert data
    channels = raw.get_data(raw.ch_names)

    # convert to microvolts to scale up precision
    channels *= 1e6

    n_channels = len(raw.ch_names)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(
            os.path.join(path, f"{filename}_leads_rereferenced{ext}"),
            n_channels=n_channels,
            file_type=FILETYPE_EDFPLUS,
        )

        channel_info = []

        ch_idx = range(n_channels)
        keys = list(raw._orig_units.keys())
        for i in ch_idx:
            try:
                ch_dict = {
                    "label": raw.ch_names[i],
                    # "dimension": raw._orig_units[keys[i]],
                    "sample_rate": raw._raw_extras[0]["n_samps"][i],
                    "physical_min": raw._raw_extras[0]["physical_min"][i],
                    "physical_max": raw._raw_extras[0]["physical_max"][i],
                    "digital_min": raw._raw_extras[0]["digital_min"][i],
                    "digital_max": raw._raw_extras[0]["digital_max"][i],
                    "transducer": "",
                    "prefilter": "",
                }
            except:
                ch_dict = {
                    "label": raw.ch_names[i],
                    # "dimension": raw._orig_units[keys[i]],
                    "sample_rate": sfreq,
                    "physical_min": channels.min(),
                    "physical_max": channels.max(),
                    "digital_min": dmin,
                    "digital_max": dmax,
                    "transducer": "",
                    "prefilter": "",
                }

            channel_info.append(ch_dict)
        f.setPatientCode(raw._raw_extras[0]["subject_info"].get("id", "0"))
        f.setPatientName(raw._raw_extras[0]["subject_info"].get("name", "noname"))
        f.setTechnician("mne-gist-save-edf-skjerns")
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(channels)
        for annotation in raw.annotations:
            onset = annotation["onset"]
            duration = annotation["duration"]
            description = annotation["description"]
            f.writeAnnotation(onset, duration, description)

    except Exception as e:
        raise e
    finally:
        print("Closing file")
        f.close()
        print("File closed")
    return True
