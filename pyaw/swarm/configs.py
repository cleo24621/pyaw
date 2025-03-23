tct16_fs = 16
mag_hr_fs = 50
mag_lr_fs = 1


class ViresSwarmRequest:
    auxiliaries = [
        "AscendingNodeLongitude",
        "QDLat",
        "QDLon",
        "QDBasis",
        "MLT",
        "SunDeclination",
    ]
    models = ["IGRF"]
    collections = {
        "EFI_TCT16": [
            "SW_EXPT_EFIA_TCT16",
            "SW_EXPT_EFIB_TCT16",
            "SW_EXPT_EFIC_TCT16",
        ],
        "MAG": ["SW_OPER_MAGA_LR_1B", "SW_OPER_MAGB_LR_1B", "SW_OPER_MAGC_LR_1B"],
        "MAG_HR": [
            "SW_OPER_MAGA_HR_1B",
            "SW_OPER_MAGB_HR_1B",
            "SW_OPER_MAGC_HR_1B",
        ],
    }