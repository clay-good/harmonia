"""SED-ML export — the simulation-protocol descriptor (pacing rate, beats to
steady state, output window) that makes a risk computation reproducible. Paired
with CellML in the cardiac-modeling convention.
"""
from __future__ import annotations

from ..load import Dataset

SEDML_NS = "http://sed-ml.org/sed-ml/level1/version3"


def build(ds: Dataset, ap_model: str = "cipaordv1.0", model_source: str = "model.cellml",
          cl: float = 2000.0, n_beats: int = 3, points_per_beat: int = 2000) -> str:
    end = cl * n_beats
    npts = points_per_beat * n_beats
    out_start = cl * (n_beats - 1)   # record the final beat
    L = ['<?xml version="1.0" encoding="UTF-8"?>',
         f'<sedML xmlns="{SEDML_NS}" level="1" version="3">',
         "  <!-- Harmonia pacing protocol: single cell, "
         f"CL={cl:g} ms ({1000.0/cl:g} Hz), {n_beats} beats to steady state, "
         "record final beat. NOT a clinical protocol. -->",
         "  <listOfModels>",
         f'    <model id="apmodel" language="urn:sedml:language:cellml" '
         f'source="{model_source}"/>',
         "  </listOfModels>",
         "  <listOfSimulations>",
         f'    <uniformTimeCourse id="pace" initialTime="0" outputStartTime="{out_start:g}" '
         f'outputEndTime="{end:g}" numberOfPoints="{npts}">',
         '      <algorithm kisaoID="KISAO:0000019"/>',  # CVODE
         "    </uniformTimeCourse>",
         "  </listOfSimulations>",
         "  <listOfTasks>",
         '    <task id="task1" modelReference="apmodel" simulationReference="pace"/>',
         "  </listOfTasks>",
         "  <listOfDataGenerators>",
         '    <dataGenerator id="time"><listOfVariables>'
         '<variable id="t" taskReference="task1" symbol="urn:sedml:symbol:time"/>'
         '</listOfVariables><math xmlns="http://www.w3.org/1998/Math/MathML">'
         '<ci>t</ci></math></dataGenerator>',
         '    <dataGenerator id="Vm"><listOfVariables>'
         '<variable id="V" taskReference="task1" target="cell/V"/>'
         '</listOfVariables><math xmlns="http://www.w3.org/1998/Math/MathML">'
         '<ci>V</ci></math></dataGenerator>',
         "  </listOfDataGenerators>",
         "  <listOfOutputs>",
         '    <plot2D id="ap_trace"><listOfCurves>'
         '<curve id="c1" logX="false" logY="false" xDataReference="time" '
         'yDataReference="Vm"/></listOfCurves></plot2D>',
         "  </listOfOutputs>",
         "</sedML>"]
    return "\n".join(L) + "\n"
