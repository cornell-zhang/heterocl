# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define HeteroCL default tool settings"""


from attr import dataclass


class Tool:
    """The base class for all device tooling
    mode (sim/impl) is decided by tool configuration
    e.g. run sw emulation by passing gcc / vivado_hls arg
    and actual impl by passing sdaccel / aocl arg
    Parameters
    ----------
    types: str
        Device of device to place data
    model: str
        Model of device to place date
    """

    def __init__(self, name, mode, kwargs):
        self.name = name
        self.mode = mode
        self.options = kwargs
        self.suported_modes = ["debug", "sw_sim", "hw_sim", "hw_exe"]

    def __getattr__(self, entry):
        return self.mapping[entry]

    # pylint: disable=dangerous-default-value
    def __call__(self, mode, setting={}):
        self.mode = mode
        self.options = setting
        return self

    def __str__(self):
        return f"{self.name}(Mode {self.mode})"

    def __repr__(self):
        return f"{self.name}(Mode {self.mode})"

    def set_mode(self, mode):
        assert (
            mode in self.suported_modes
        ), f"{mode} not supported {self.suported_modes}"
        self.mode = mode


class HLS(Tool):
    def __init__(self, name="vivado_hls"):
        mode = "sw_sim"
        if name == "vivado_hls":
            options = {"Frequency": "100", "Version": "2019.2"}
        else:  # vitis_hls
            options = {"Frequency": "300", "Version": "2020.2"}
        super().__init__(name, mode, options)
        self.suported_modes = ["debug", "custom", "csim", "csyn", "cosim", "impl"]

    def set_mode(self, mode):
        if mode not in {"custom", "debug"}:
            input_modes = mode.split("|")
            modes = ["csim", "csyn", "cosim", "impl"]
            new_modes = []
            for in_mode in input_modes:
                assert in_mode in modes, "supported tool mode: " + str(modes)
                # check validity, dependency shown below
                # csim (opt) -\    /- cosim
                #              |--|
                #    csyn    -/    \- impl
                if in_mode in {"cosim", "impl"}:
                    new_modes.append("csyn")
                    print(
                        f"Warning: csyn needs to be done before {in_mode}, so csyn is added to target mode."
                    )
                new_modes.append(in_mode)
            mode = list(set(new_modes))
            mode.sort()
            new_mode = "|".join(mode)
        else:
            new_mode = mode
        self.mode = new_mode


class Vitis(Tool):
    def __init__(self):
        name = "vitis"
        mode = "sw_sim"
        options = {"Frequency": "300", "Version": "2019.2"}
        super().__init__(name, mode, options)

        self.tool_mode = None
        self.xpfm = None
        self.binary = None
        self.build_dir = None


class SDAccel(Vitis):
    pass


class AOCL(Tool):
    def __init__(self):
        name = "aocl"
        mode = "sw_sim"
        options = {"Frequency": "500", "Version": "19.2"}
        super().__init__(name, mode, options)


@dataclass
class OptionTable:
    llvm = ("sw_sim", {"version": "6.0.0"})
    sdaccel = ("sw_sim", {"version": "2017.1", "clock": "1"})
    sdsoc = ("sw_sim", {"version": "2017.1", "clock": "1"})
    vitis = ("sw_sim", {"version": "2019.2", "clock": "1"})
    vivado_hls = ("sw_sim", {"version": "2017.1"})
    rocket = ("debug", {"RISCV": ""})
    # refer to xilinx2016_1/ug904-vivado-implementation.pdf
    vivado = (
        "pnr",
        {
            "version": "2017.1",
            "logic": [
                "Default",
                "Explore",
                "ExploreSequentialArea",
                "AddRemap",
                "ExploreArea",
            ],
            "placement": [
                "Default",
                "Explore",
                "ExtraNetDelay_high",
                "ExtraNetDelay_medium",
                "ExtraNetDelay_low",
                "ExtraPostPlacementOpt",
                "WLDrivenBlockPlacement",
                "LateBlockPlacement",
                "AltSpreadLogic_low",
                "AltSpreadLogic_medium",
                "AltSpreadLogic_high",
            ],
            "routing": ["Default", "Explore", "HigherDelayCost"],
            "fanout_opt": ["on", "off"],
            "placement_opt": ["on", "off"],
            "critical_cell_opt": ["on", "off"],
            "critical_pin_opt": ["on", "off"],
            "retime": ["on", "off"],
            "rewire": ["on", "off"],
        },
    )
    quartus = (
        "pnr",
        {
            "version": "17.1",
            "auto_dsp_recognition": ["On", "Off"],
            "disable_register_merging_across_hierarchies": ["On", "Off", "Auto"],
            "mux_restructure": ["On", "Off", "Auto"],
            "optimization_technique": ["Area", "Speed", "Balanced"],
            "synthesis_effort": ["Auto", "Fast"],
            "synth_timing_driven_synthesis": ["On", "Off"],
            "fitter_aggressive_routability_optimization": [
                "Always",
                "Automatically",
                "Never",
            ],
            "fitter_effort": ["Standard Fit", "Auto Fit"],
            "remove_duplicate_registers": ["On", "Off"],
            "adv_netlist_opt_synth_wysiwyg_remap": ["On", "Off"],
            "allow_any_ram_size_for_recognition": ["On", "Off"],
            "allow_any_rom_size_for_recognition": ["On", "Off"],
            "allow_any_shift_register_size_for_recognition": ["On", "Off"],
            "allow_power_up_dont_care": ["On", "Off"],
            "allow_shift_register_merging_across_hierarchies": [
                "Always",
                "Auto",
                "Off",
            ],
            "allow_synch_ctrl_usage": ["On", "Off"],
            "auto_carry_chains": ["On", "Off"],
            "auto_clock_enable_recognition": ["On", "Off"],
            "auto_enable_smart_compile": ["On", "Off"],
            "auto_open_drain_pins": ["On", "Off"],
            "auto_ram_recognition": ["On", "Off"],
            "auto_resource_sharing": ["On", "Off"],
            "auto_rom_recognition": ["On", "Off"],
            "auto_shift_register_recognition": ["Always", "Auto", "Off"],
            "enable_state_machine_inference": ["On", "Off"],
            "force_synch_clear": ["On", "Off"],
            "ignore_carry_buffers": ["On", "Off"],
            "ignore_cascade_buffers": ["On", "Off"],
            "ignore_max_fanout_assignments": ["On", "Off"],
            "infer_rams_from_raw_logic": ["On", "Off"],
            "optimize_power_during_synthesis": [
                "Extra effort",
                "Normal compilation",
                "Off",
            ],
            "shift_register_recognition_aclr_signal": ["On", "Off"],
            "state_machine_processing": [
                "Auto",
                "Gray",
                "Johnson, Minimal Bits",
                "One-Hot",
                "Sequential",
                "User-Encoded",
            ],
            "strict_ram_recognition": ["On", "Off"],
            "synthesis_keep_synch_clear_preset_behavior_in_unmapper": ["On", "Off"],
            "synth_resource_aware_inference_for_block_ram": ["On", "Off"],
            "alm_register_packing_effort": ["High", "Low", "Medium"],
            "auto_delay_chains": ["On", "Off"],
            "auto_delay_chains_for_high_fanout_input_pins": ["On", "Off"],
            "eco_optimize_timing": ["On", "Off"],
            "final_placement_optimization": ["Always", "Automatically", "Never"],
            "optimize_for_metastability": ["On", "Off"],
            "optimize_hold_timing": [
                "All Paths",
                "IO Paths and Minimum TPD Paths",
                "Off",
            ],
            "optimize_ioc_register_placement_for_timing": [
                "Normal",
                "Off",
                "Pack All IO Registers",
            ],
            "optimize_multi_corner_timing": ["On", "Off"],
            "optimize_power_during_fitting": [
                "Extra effort",
                "Normal compilation",
                "Off",
            ],
            "physical_synthesis": ["On", "Off"],
            "placement_effort_multiplier": [0.2, 0.5, 1.0, 2.0, 3.0, 4.0],
            "programmable_power_technology_setting": [
                "Automatic",
                "Force All Tiles with Failing Timing Paths to High Speed",
                "Force All Used Tiles to High Speed",
                "Minimize Power Only",
            ],
            "qii_auto_packed_registers": [
                "Auto",
                "Minimize Area",
                "Minimize Area with Chains",
                "Normal",
                "Off",
                "Sparse",
                "Sparse Auto",
            ],
            "router_clocking_topology_analysis": ["On", "Off"],
            "router_lcell_insertion_and_logic_duplication": ["Auto", "On", "Off"],
            "router_register_duplication": ["Auto", "On", "Off"],
            "router_timing_optimization_level": ["MINIMUM", "Normal", "MAXIMUM"],
            "seed": (1, 5),
            "tdc_aggressive_hold_closure_effort": ["On", "Off"],
            "allow_register_retiming": ["On", "Off"],
        },
    )
    aocl = ("sw_sim", {"version": "17.0", "clock": "1.5"})


Tool.vivado_hls = HLS("vivado_hls")
Tool.vitis_hls = HLS("vitis_hls")
Tool.vitis = Vitis()
Tool.aocl = AOCL()
Tool.sdaccel = SDAccel()
