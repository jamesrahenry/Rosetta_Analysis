#!/usr/bin/env python3
"""G5b — random-text calibration null at n=500, ORIGINAL corpus (P4, tbc29f76 item 3, take 2).

G5 (g5_random_text_null.py) reran the random-text null at n=500 but had to
substitute a new corpus (wikitext-103) because the original 200-text corpus
behind the published 0.1484 figure lived only in the external analysis repo
and wasn't checked out on that session's host. That corpus swap turned out
to dominate the result more than the sample-size increase it was meant to
isolate (0.1484 -> 0.2291 at matched n=200 on the new corpus alone), which
is not a clean test of the size-matching question. See
papers/prh-validation/preprint.md §4.5 "Random-text null calibration size"
and the P4 review session that flagged this, 2026-07-17.

This job does it properly: the ORIGINAL 201-sentence corpus was recovered
from jamesrahenry/Rosetta_Analysis (public repo, alignment/align_random_calib.py,
RANDOM_CORPUS) -- not lost, just not present in this checkout. It is embedded
below verbatim (ORIGINAL_CORPUS_201), extended with 299 new sentences in the
exact same documented style (EXTENSION_CORPUS_299 -- same topic list, same
explicit avoidance of contrastive/causal/moral/temporal language that would
activate the 17 concept axes) to reach n=500, matching the primary analysis's
calibration size.

Stage B reports THREE numbers, not two:
  - aligned_cos_n200_original_subsample: the EXACT original sampling method
    (np.random.default_rng(seed=42).shuffle() on ORIGINAL_CORPUS_201, first
    200) -- this is the direct, byte-identical bridge to the published 0.1484
    figure. If the pipeline is faithful, this should closely reproduce it.
  - aligned_cos_n500: full 500-sentence corpus (201 original + 299 extension),
    fixed order -- the size-matched, corpus-continuous result.
  - raw_cos: pre-rotation baseline, as in G5.

Stage A (GPU, per model): extract last-token activations for the 500 texts at
every layer; save one .npz shard per model, upload to HF.
Stage B (no GPU): for every ordered cross-family same-dimension pair in the
alignment roster (clusters A-E) x 17 concepts: mean-center both models'
random-text matrices at their own concept-peak layers, fit R
(scipy orthogonal_procrustes, float64, zero-PCA/same-dim), report
cos(dom_src, dom_tgt @ R) at n=500 and at the original n=200 subsample.

PIGGYBACK JOB, not gating anything: run on the exfil-rerun session's host
during idle windows (same pattern as the gemma-2-27b stability check,
tfa2acf7). Does not need its own host spin-up -- reuses whichever host is
already loading the alignment roster models.

Written: 2026-07-17 UTC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import (
    CKPT_ROOT, CONCEPTS_17, OUT_ROOT, alignment_roster_from_hf, dom_matrix,
    family_of, hf_upload, hf_verify, load_caz, log, peak_layer, shard_done,
    shard_write,
)
from forward_utils import calibrate_offset, load_model, plain_acts, release

JOB = "g5b"
N_SUB = 200
SEED = 42
ACTS_DIR = CKPT_ROOT / JOB / "acts"
TEXTS_FILE = CKPT_ROOT / JOB / "neutral_texts.json"


# ---------------------------------------------------------------------------
# Neutral corpus — recovered original (201) + extension (299) = 500
# ---------------------------------------------------------------------------

ORIGINAL_CORPUS_201 = [
    "The capital of Australia is Canberra, not Sydney.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Amazon River discharges more water than any other river on Earth.",
    "A standard chess board has 64 squares arranged in an 8×8 grid.",
    "Mount Everest stands at approximately 8,849 metres above sea level.",
    "The human body contains roughly 37 trillion cells.",
    "Sourdough bread is leavened using a fermented flour-and-water starter.",
    "Mercury is the closest planet to the Sun in our solar system.",
    "The Eiffel Tower was completed in 1889 as a temporary exhibition structure.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "A haiku traditionally consists of three lines with 5, 7, and 5 syllables.",
    "The Great Wall of China stretches over 21,000 kilometres in total.",
    "Fibonacci numbers appear frequently in the spiral patterns of sunflowers.",
    "The Pacific Ocean covers more area than all of Earth's landmasses combined.",
    "A standard marathon is 42.195 kilometres, or 26.2 miles.",
    "Penguins are found exclusively in the Southern Hemisphere.",
    "The periodic table currently contains 118 confirmed chemical elements.",
    "A kilogram is defined by the Planck constant, not a physical artefact.",
    "The Sahara Desert occupies roughly 9 million square kilometres of Africa.",
    "Bamboo is technically classified as a grass, not a tree.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "The International Space Station orbits at about 400 kilometres altitude.",
    "Honey has an indefinite shelf life due to its low water content and acidity.",
    "The human genome contains approximately 3 billion base pairs.",
    "Tokyo is the most populous metropolitan area in the world.",
    "A standard basketball court measures 28 metres by 15 metres.",
    "The moon's diameter is roughly a quarter of Earth's diameter.",
    "Avocados are botanically classified as a single-seeded berry.",
    "Glass is an amorphous solid, not a slow-flowing liquid.",
    "The wingspan of an albatross can reach up to 3.5 metres.",
    "Copper is one of the few metals that occurs in its native form in nature.",
    "The Dead Sea is approximately 430 metres below sea level.",
    "Most commercial aircraft cruise at altitudes between 9,000 and 12,000 metres.",
    "A square kilometre equals one million square metres.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Hydrogen is the most abundant element in the observable universe.",
    "An octopus has three hearts and blue blood.",
    "The Louvre museum in Paris houses more than 35,000 works of art.",
    "Pluto was reclassified as a dwarf planet in 2006.",
    "The human eye can distinguish approximately 10 million different colours.",
    "A standard piano keyboard has 88 keys.",
    "Lake Superior is the largest of the five Great Lakes by surface area.",
    "The wingspan of a Boeing 747 is approximately 68 metres.",
    "Antarctica holds about 70 percent of Earth's fresh water as ice.",
    "The Andes mountain range runs along the western edge of South America.",
    "The average depth of the ocean is about 3,688 metres.",
    "A light-year is the distance light travels in one year, about 9.46 trillion kilometres.",
    "The Colosseum in Rome was completed around 80 AD.",
    "Salt water freezes at a lower temperature than fresh water.",
    "The diameter of a standard football goal is 7.32 metres wide and 2.44 metres tall.",
    "Venus rotates in the opposite direction to most planets.",
    "Platinum is denser than gold.",
    "The human liver performs over 500 distinct biochemical functions.",
    "The tallest tree species in the world is the coast redwood.",
    "A full moon appears about 14 percent larger at perigee than at apogee.",
    "The Mariana Trench reaches a depth of approximately 11 kilometres.",
    "Table salt is a compound of sodium and chloride ions.",
    "The wingspan of the wandering albatross averages around 300 centimetres.",
    "Rubber is derived from the latex sap of the Hevea brasiliensis tree.",
    "The Nile is generally considered the longest river, at about 6,650 kilometres.",
    "A honeybee visits around 50 to 100 flowers during a single foraging trip.",
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars.",
    "The freezing point of ethanol is −114.1 degrees Celsius.",
    "The Suez Canal connects the Mediterranean Sea to the Red Sea.",
    "An adult blue whale can weigh up to 180 metric tonnes.",
    "The Rosetta Stone was discovered in Egypt in 1799.",
    "Granite is an igneous rock composed primarily of quartz, feldspar, and mica.",
    "The cornea is the transparent front layer of the eye.",
    "The wingspan of a monarch butterfly is between 8.6 and 12.4 centimetres.",
    "A standard shipping container is 6 or 12 metres long.",
    "Carbon dioxide makes up about 0.04 percent of Earth's atmosphere.",
    "The Taj Mahal was built between 1631 and 1648.",
    "Dolphins use echolocation to navigate and hunt in murky water.",
    "The surface temperature of the Sun is approximately 5,500 degrees Celsius.",
    "A standard wine bottle holds 750 millilitres.",
    "The brain of an adult human weighs approximately 1.4 kilograms.",
    "The deepest lake in the world is Lake Baikal in Siberia.",
    "Monarch butterflies migrate up to 4,800 kilometres each year.",
    "The chemical symbol for gold is Au, from the Latin word aurum.",
    "The Parthenon in Athens was constructed between 447 and 432 BC.",
    "A standard USB 3.0 port transfers data at up to 5 gigabits per second.",
    "The Atacama Desert in Chile is one of the driest places on Earth.",
    "A square metre of office paper holds roughly 500 sheets.",
    "The wingspan of a wandering albatross can exceed three metres.",
    "The circumference of Earth at the equator is approximately 40,075 kilometres.",
    "Seawater is about 3.5 percent dissolved salts by mass.",
    "The Pythagorean theorem states that a² + b² = c² for right-angled triangles.",
    "Gold has an atomic number of 79.",
    "The Hubble Space Telescope was launched in April 1990.",
    "A standard brick measures roughly 215 × 102.5 × 65 millimetres.",
    "The blue whale is the largest animal known to have ever existed.",
    "Oxygen makes up approximately 21 percent of Earth's atmosphere.",
    "The International Date Line runs roughly along the 180th meridian.",
    "Silicon is the second most abundant element in Earth's crust.",
    "The wingspan of a peregrine falcon is between 74 and 120 centimetres.",
    "Lake Victoria is the largest lake in Africa by surface area.",
    "The Burj Khalifa stands at 828 metres, making it the tallest building.",
    "A standard football pitch is between 100 and 110 metres long.",
    "Obsidian is a naturally occurring volcanic glass.",
    "The human skeleton consists of 206 bones in an adult.",
    "The wavelength of visible light ranges from about 380 to 700 nanometres.",
    "Chile is the world's longest country from north to south.",
    "A standard A4 sheet measures 210 × 297 millimetres.",
    "The cheetah is the fastest land animal, reaching speeds of up to 112 km/h.",
    "The Mississippi River drains about 40 percent of the continental United States.",
    "Diamonds are the hardest natural material, rating 10 on the Mohs scale.",
    "The orbit of the Moon around Earth takes approximately 27.3 days.",
    "The thermal conductivity of copper is about 400 watts per metre-kelvin.",
    "Iceland is the most volcanically active country in the world per unit area.",
    "The wingspan of a bald eagle ranges from 183 to 244 centimetres.",
    "A standard Olympic swimming pool holds 2,500,000 litres of water.",
    "The chemical formula for water is H₂O.",
    "The Gobi Desert spans parts of China and Mongolia.",
    "A standard deck of playing cards contains 52 cards in 4 suits.",
    "The tallest building in the world as of 2024 is the Burj Khalifa.",
    "Stalactites grow down from cave ceilings; stalagmites grow up from floors.",
    "The Coriolis effect deflects moving objects to the right in the Northern Hemisphere.",
    "A kilobyte is 1,024 bytes; a megabyte is 1,024 kilobytes.",
    "The wingspan of an emperor penguin's flippers is around 30 centimetres.",
    "The density of iron at room temperature is approximately 7.87 g/cm³.",
    "The Amazon rainforest produces about 20 percent of the world's oxygen.",
    "A standard tennis court is 23.77 metres long and 10.97 metres wide.",
    "The Hoover Dam spans the Colorado River on the border of Nevada and Arizona.",
    "The boiling point of nitrogen is −195.8 degrees Celsius.",
    "The Sahel is a semi-arid transition zone between the Sahara and the savanna.",
    "The wingspan of a golden eagle ranges from 180 to 234 centimetres.",
    "Concrete is made from cement, water, sand, and aggregate.",
    "The human ear can detect sounds between 20 and 20,000 hertz.",
    "The Drake Passage separates South America from Antarctica.",
    "A standard brick wall uses about 48 bricks per square metre of wall area.",
    "The wingspan of a California condor can reach up to 3 metres.",
    "The melting point of iron is approximately 1,538 degrees Celsius.",
    "The Bering Strait separates Russia from Alaska at its narrowest point.",
    "A standard SD card uses NAND flash memory.",
    "The wingspan of a harpy eagle can reach up to 224 centimetres.",
    "The Himalayas were formed by the collision of the Indian and Eurasian plates.",
    "A standard rugby union pitch is between 94 and 100 metres long.",
    "The speed of sound in air at 20°C is approximately 343 metres per second.",
    "The Strait of Gibraltar connects the Atlantic Ocean to the Mediterranean Sea.",
    "Tungsten has the highest melting point of any element, at 3,422 degrees Celsius.",
    "The wingspan of a barn owl averages about 85 centimetres.",
    "The Panama Canal is 82 kilometres long.",
    "Aluminum is the most abundant metal in Earth's crust.",
    "A standard golf hole is 108 millimetres in diameter.",
    "The wingspan of a swan ranges from 200 to 238 centimetres.",
    "The Ganges River flows through India and Bangladesh.",
    "A standard ship anchor for a large vessel weighs between 10 and 20 tonnes.",
    "The wingspan of a great horned owl is between 91 and 153 centimetres.",
    "The Caspian Sea is the world's largest inland body of water.",
    "A standard soccer ball has a circumference of 68 to 70 centimetres.",
    "The wingspan of a flamingo is between 95 and 100 centimetres.",
    "The Congo River has the second-largest flow by volume of any river.",
    "A standard violin has four strings tuned to G, D, A, and E.",
    "The wingspan of a red-tailed hawk ranges from 114 to 133 centimetres.",
    "The Lena River is one of the longest rivers in the world.",
    "A standard bicycle wheel has between 28 and 36 spokes.",
    "The wingspan of a snowy owl is between 125 and 150 centimetres.",
    "The Baltic Sea has a lower salt content than typical ocean water.",
    "A standard baseball diamond has 90-foot base paths.",
    "The wingspan of a white stork is between 155 and 215 centimetres.",
    "The Columbia River forms most of the border between Oregon and Washington.",
    "A standard guitar has six strings tuned E, A, D, G, B, E.",
    "The wingspan of a black vulture is between 132 and 150 centimetres.",
    "The Yellow River is the second-longest river in China.",
    "A standard stopwatch can measure time to one-hundredth of a second.",
    "The wingspan of a grey heron ranges from 155 to 175 centimetres.",
    "The Tigris and Euphrates rivers flow through the Mesopotamian plain.",
    "A standard car tyre pressure is between 30 and 35 psi.",
    "The wingspan of a pelican can be up to 360 centimetres.",
    "The Ural Mountains form the conventional boundary between Europe and Asia.",
    "A standard microwave oven operates at 2.45 gigahertz.",
    "The wingspan of a turkey vulture ranges from 160 to 183 centimetres.",
    "The Rhine River flows through Switzerland, Germany, and the Netherlands.",
    "A standard commercial airliner has between 100 and 550 passenger seats.",
    "The wingspan of a secretary bird can reach up to 207 centimetres.",
    "The Volga River is the longest river in Europe.",
    "A standard incandescent light bulb converts about 5 percent of energy to light.",
    "The wingspan of a steller's sea eagle can reach up to 250 centimetres.",
    "The Danube River flows through 10 countries in Europe.",
    "A standard swimming pool lap is 50 metres in competition pools.",
    "The wingspan of a common buzzard ranges from 109 to 136 centimetres.",
    "The Murray-Darling river system is the longest in Australia.",
    "A standard football weighs between 410 and 450 grams.",
    "The wingspan of an osprey ranges from 127 to 180 centimetres.",
    "The Zambezi River in Africa flows over Victoria Falls.",
    "A standard basketball weighs between 567 and 624 grams.",
    "The wingspan of a short-eared owl is between 85 and 110 centimetres.",
    "The Rhine delta empties into the North Sea in the Netherlands.",
    "A standard sprinting track is 400 metres per lap.",
    "The wingspan of a peregrine falcon averages about 95 centimetres.",
    "The Niger River flows through Guinea, Mali, Niger, and Nigeria.",
    "A standard cricket pitch is 20.12 metres between wickets.",
    "The wingspan of a lanner falcon is between 95 and 105 centimetres.",
    "The Mekong River flows through China, Myanmar, Laos, Thailand, Cambodia, and Vietnam.",
    "A standard badminton court is 13.4 metres long and 6.1 metres wide.",
    "The wingspan of a common raven is between 115 and 150 centimetres.",
    "The Irrawaddy River is the principal river of Myanmar.",
    "A standard squash court is 9.75 metres long and 6.4 metres wide.",
    "The wingspan of a black kite averages about 140 centimetres.",
    "The Orinoco River drains a large part of Venezuela and Colombia.",
    "A standard handball court is 40 metres long and 20 metres wide.",
]

EXTENSION_CORPUS_299 = [
    "The Sydney Opera House has a roof composed of over one million ceramic tiles.",
    "A standard Rubik's Cube has 43 quintillion possible configurations.",
    "The blue-ringed octopus is native to tide pools in the Pacific and Indian Oceans.",
    "A tablespoon holds approximately 15 millilitres of liquid.",
    "The Karakoram range contains five of the world's fourteen 8,000-metre peaks.",
    "Baking soda is sodium bicarbonate, a common leavening agent.",
    "A standard hockey rink measures 61 metres by 30.5 metres.",
    "The largest known prime number has more than 24 million digits.",
    "The Byzantine Empire's capital, Constantinople, fell in 1453.",
    "A standard shoe size chart differs between UK, US, and EU systems.",
    "Server racks are typically measured in rack units, each 44.45 millimetres tall.",
    "Barometric pressure at sea level averages about 1,013 hectopascals.",
    "The saguaro cactus can live for more than 150 years.",
    "A standard teaspoon holds approximately 5 millilitres.",
    "The Strait of Malacca is one of the busiest shipping lanes in the world.",
    "Quinoa is technically a seed, not a true cereal grain.",
    "A regulation volleyball net stands 2.43 metres high for men's play.",
    "Euclid's Elements consists of thirteen books on geometry.",
    "The Ottoman Empire lasted from 1299 to 1922.",
    "A standard bowling lane is 18.29 metres long.",
    "Fibre-optic cables transmit data using pulses of light.",
    "The dew point is the temperature at which air becomes saturated with moisture.",
    "The kiwi fruit originated in China before cultivation spread to New Zealand.",
    "A standard cup in US recipes equals 236.6 millilitres.",
    "The Strait of Hormuz connects the Persian Gulf to the Gulf of Oman.",
    "Cork is harvested from the bark of the cork oak tree.",
    "A regulation table tennis table is 2.74 metres long.",
    "The number pi has been calculated to over 100 trillion digits.",
    "The Mughal Empire governed much of the Indian subcontinent from 1526 to 1857.",
    "A standard piano has 52 white keys and 36 black keys.",
    "Ethernet cables come in categories such as Cat5e, Cat6, and Cat6a.",
    "A cumulonimbus cloud can extend upward more than 12 kilometres.",
    "Saffron comes from the dried stigmas of the crocus flower.",
    "A regulation curling sheet is 44.5 metres long.",
    "The Cambrian explosion occurred roughly 541 million years ago.",
    "The Han dynasty ruled China for over four centuries.",
    "A standard 35mm film frame measures 36 by 24 millimetres.",
    "Graphene is a single layer of carbon atoms arranged in a hexagonal lattice.",
    "The Skagerrak strait separates Norway from Denmark.",
    "Miso paste is made by fermenting soybeans with koji mould.",
    "A regulation rugby league field is 100 metres long between try lines.",
    "Archimedes calculated an approximation of pi using inscribed polygons.",
    "The Sasanian Empire preceded the Islamic conquest of Persia.",
    "A standard 3.5mm audio jack has become the most common analog connector.",
    "The tropopause marks the boundary between the troposphere and stratosphere.",
    "Tempering chocolate stabilizes its cocoa butter crystals for a glossy finish.",
    "A regulation badminton shuttlecock has 16 feathers.",
    "The Fields Medal is awarded to mathematicians under the age of 40.",
    "The Qin dynasty unified China in 221 BC.",
    "A standard VGA connector has 15 pins arranged in three rows.",
    "Cirrus clouds form at altitudes above 6,000 metres.",
    "Kombucha is produced by fermenting sweetened tea with a bacterial culture.",
    "A regulation Australian rules football oval can be up to 185 metres long.",
    "The golden ratio is approximately 1.618.",
    "The Aksumite Empire controlled trade routes in the Horn of Africa.",
    "A standard HDMI cable supports both audio and video signals.",
    "Altocumulus clouds often appear in patchy, rippled formations.",
    "Tahini is made by grinding hulled sesame seeds into a paste.",
    "A regulation lacrosse field is 110 metres long.",
    "Prime numbers greater than 3 are always of the form 6n±1.",
    "The Maurya Empire once covered nearly the entire Indian subcontinent.",
    "A standard SATA cable transfers data at up to 6 gigabits per second.",
    "Stratus clouds typically form a featureless grey layer close to the ground.",
    "Yeast converts sugars into carbon dioxide and alcohol during fermentation.",
    "A regulation field hockey pitch is 91.4 metres long.",
    "A rhombus is a quadrilateral with four equal sides.",
    "The Khmer Empire built the temple complex of Angkor Wat.",
    "A standard Bluetooth connection operates in the 2.4 gigahertz band.",
    "Nimbostratus clouds are associated with continuous, steady rainfall.",
    "Ghee is clarified butter with the milk solids removed.",
    "A regulation water polo pool is 30 metres long.",
    "A trapezoid has exactly one pair of parallel sides.",
    "The Inca road system stretched over 30,000 kilometres across South America.",
    "A standard RJ45 connector is used for wired network connections.",
    "Fog forms when air near the ground cools to its dew point.",
    "Miso soup traditionally includes dashi stock as its base.",
    "A regulation squash ball must bounce to a specific height at 45 degrees Celsius.",
    "A regular hexagon has interior angles of 120 degrees each.",
    "The Kingdom of Kush flourished along the Nile south of Egypt.",
    "A standard microSD card is 15 by 11 by 1 millimetres.",
    "Hailstones form when raindrops are carried upward repeatedly by strong updrafts.",
    "Sourdough starters rely on a symbiotic culture of yeast and bacteria.",
    "A regulation javelin for men weighs 800 grams.",
    "The sum of angles in any triangle is always 180 degrees.",
    "The Toltec civilization preceded the rise of the Aztec Empire.",
    "A standard PCIe slot can support several different lane configurations.",
    "Virga is precipitation that evaporates before reaching the ground.",
    "Umami is often described as the fifth basic taste, alongside sweet and sour.",
    "A regulation discus for men weighs two kilograms.",
    "An isosceles triangle has at least two sides of equal length.",
    "The Zapotec civilization built the city of Monte Albán.",
    "A standard USB-C connector is reversible, unlike earlier USB types.",
    "A rainbow forms when sunlight is refracted and reflected inside raindrops.",
    "Tofu is made by coagulating soy milk and pressing the curds into blocks.",
    "A regulation shot put for men weighs 7.26 kilograms.",
    "A parallelogram's opposite sides are equal in length and parallel.",
    "The Nazca Lines are large geoglyphs etched into the desert of southern Peru.",
    "A standard NVMe drive connects via the PCIe interface for faster data transfer.",
    "Sleet forms when snowflakes partially melt and refreeze before landing.",
    "Kimchi is traditionally fermented using napa cabbage and Korean chili flakes.",
    "A regulation hammer throw weighs 7.26 kilograms for men's competition.",
    "A rectangle's diagonals are always equal in length.",
    "The Olmec civilization is considered the earliest major Mesoamerican culture.",
    "A standard Wi-Fi router typically broadcasts on the 2.4 and 5 gigahertz bands.",
    "A derecho is a widespread, long-lived windstorm associated with thunderstorms.",
    "Pickling preserves vegetables by submerging them in an acidic brine.",
    "A regulation pole vault bar can be raised in increments as small as one centimetre.",
    "An equilateral triangle has three sides of equal length and three 60-degree angles.",
    "The Chavín culture predates the Inca by roughly two thousand years.",
    "A standard optical mouse uses light to detect movement across a surface.",
    "A waterspout is a tornado that forms over a body of water.",
    "Brining meat before cooking helps it retain moisture.",
    "A regulation long jump pit is filled with sand for athlete safety.",
    "A cube has six faces, twelve edges, and eight vertices.",
    "The Moche civilization is known for its detailed ceramic pottery.",
    "A standard mechanical keyboard uses individual switches beneath each key.",
    "A microburst is a localized column of sinking air from a thunderstorm.",
    "Braising combines searing meat and then slow-cooking it in liquid.",
    "A regulation triple jump involves a hop, a step, and a jump phase.",
    "A sphere has a constant radius from its center to every point on its surface.",
    "The Wari civilization predates the Inca Empire in the Andes.",
    "A standard graphics card connects to a motherboard via a PCIe x16 slot.",
    "A supercell is a thunderstorm characterized by a rotating updraft.",
    "Poaching cooks food gently in liquid kept just below a simmer.",
    "A regulation steeplechase race includes water jump obstacles.",
    "A cylinder's volume is calculated as pi times radius squared times height.",
    "The Tiwanaku civilization was centered near Lake Titicaca.",
    "A standard solid-state drive has no moving mechanical parts.",
    "A katabatic wind is caused by cold, dense air flowing downhill.",
    "Deglazing a pan involves adding liquid to loosen browned bits of food.",
    "A regulation decathlon consists of ten track and field events.",
    "A cone's lateral surface area depends on its radius and slant height.",
    "The Chimu kingdom built the adobe city of Chan Chan in Peru.",
    "A standard motherboard chipset manages communication between components.",
    "A chinook wind can raise temperatures rapidly on the leeward side of mountains.",
    "Blanching vegetables briefly in boiling water preserves their color.",
    "A regulation heptathlon is contested over two days.",
    "A regular pentagon has interior angles of 108 degrees.",
    "The Zhou dynasty was the longest-ruling dynasty in Chinese history.",
    "A standard CPU cooler dissipates heat using a fan and heatsink.",
    "The foehn effect describes warm, dry winds descending a mountain range.",
    "Sous vide cooking holds food at a precise, low temperature in a water bath.",
    "A regulation cricket ball weighs between 155.9 and 163 grams.",
    "A regular octagon has interior angles of 135 degrees.",
    "The Xia dynasty is traditionally considered China's first dynasty.",
    "A standard power supply unit converts AC mains power to DC voltage.",
    "A monsoon is a seasonal reversal of prevailing wind patterns.",
    "Caramelizing sugar involves heating it until it browns and develops flavor.",
    "A regulation snooker table is 3.57 metres long.",
    "A regular dodecagon has twelve equal sides and twelve equal angles.",
    "The Shang dynasty is the earliest Chinese dynasty confirmed by archaeology.",
    "A standard RAM module for desktops uses the DIMM form factor.",
    "An anticyclone is an area of high atmospheric pressure.",
    "Fermented fish sauce is a staple condiment in much of Southeast Asian cooking.",
    "A regulation darts board hangs with its center 1.73 metres from the floor.",
    "A rhombus's diagonals bisect each other at right angles.",
    "The Joseon dynasty ruled Korea for roughly five centuries.",
    "A standard laptop battery is typically a lithium-ion or lithium-polymer cell.",
    "A trade wind blows consistently from east to west near the equator.",
    "Curing meat with salt draws out moisture and inhibits bacterial growth.",
    "A regulation billiards table is slightly larger than a standard pool table.",
    "A kite quadrilateral has two pairs of adjacent equal sides.",
    "The Goryeo dynasty gave Korea its modern English name.",
    "A standard external hard drive connects via USB or Thunderbolt.",
    "A polar vortex is a large area of low pressure near the Earth's poles.",
    "Rendering fat slowly separates it from connective tissue for use in cooking.",
    "A regulation archery target has ten concentric scoring rings.",
    "A scalene triangle has three sides of different lengths.",
    "The Silla kingdom unified much of the Korean peninsula in the 7th century.",
    "A standard webcam typically captures video at 30 or 60 frames per second.",
    "A jet stream is a narrow band of strong winds high in the atmosphere.",
    "Emulsifying oil and vinegar produces a stable vinaigrette.",
    "A regulation fencing piste is 14 metres long.",
    "An obtuse triangle has one interior angle greater than 90 degrees.",
    "The Baekje kingdom was one of the Three Kingdoms of ancient Korea.",
    "A standard flash drive uses NAND flash memory similar to an SD card.",
    "An occluded front forms when a cold front overtakes a warm front.",
    "Reducing a sauce concentrates its flavor by evaporating liquid.",
    "A regulation judo mat area is at least 8 metres square.",
    "An acute triangle has all three interior angles less than 90 degrees.",
    "The Parthian Empire controlled much of ancient Persia for centuries.",
    "A standard router firmware can usually be updated over the network.",
    "A squall line is a row of thunderstorms along or ahead of a cold front.",
    "Marinating meat in acid can help tenderize its texture before cooking.",
    "A regulation taekwondo mat measures 12 metres by 12 metres.",
    "A right triangle has one interior angle equal to exactly 90 degrees.",
    "The Seleucid Empire emerged after the death of Alexander the Great.",
    "A standard smartphone display uses either an LCD or OLED panel.",
    "A haboob is a large dust storm caused by a thunderstorm's downdraft.",
    "Proofing bread dough allows yeast to produce carbon dioxide before baking.",
    "A regulation wrestling mat has a circular scoring area at its center.",
    "A quadrilateral's interior angles always sum to 360 degrees.",
    "The Achaemenid Empire was founded by Cyrus the Great.",
    "A standard tablet device typically weighs between 400 and 700 grams.",
    "A whiteout occurs when falling or blowing snow reduces visibility to near zero.",
    "Zesting citrus fruit extracts aromatic oils from the peel.",
    "A regulation boxing ring is between 4.9 and 6.1 metres per side.",
    "The sum of a polygon's exterior angles is always 360 degrees.",
    "The Ptolemaic dynasty ruled Egypt following Alexander's conquest.",
    "A standard e-reader display uses electronic ink technology.",
    "A thermal is a rising column of warm air used by soaring birds and gliders.",
    "Whisking egg whites incorporates air to create a stable foam.",
    "A regulation gymnastics floor exercise area is 12 metres square.",
    "A regular heptagon has interior angles of approximately 128.6 degrees.",
    "The Nabataean kingdom carved the city of Petra into sandstone cliffs.",
    "A standard drone typically uses four or more rotors for lift.",
    "A land breeze blows from land to sea during the night.",
    "Reducing heat gradually prevents dairy-based sauces from curdling.",
    "A regulation trampoline bed measures 4.28 by 2.14 metres.",
    "A concave polygon has at least one interior angle greater than 180 degrees.",
    "The Hittite Empire controlled much of Anatolia in the Bronze Age.",
    "A standard smartwatch battery typically lasts one to two days per charge.",
    "A sea breeze develops when land heats faster than the adjacent ocean.",
    "Toasting spices before grinding intensifies their aroma.",
    "A regulation weightlifting platform measures four metres square.",
    "A convex polygon has no interior angle greater than 180 degrees.",
    "The Assyrian Empire was known for its extensive network of roads.",
    "A standard mesh Wi-Fi system uses multiple nodes to extend coverage.",
    "Convective precipitation forms from rapidly rising, unstable air.",
    "Simmering, unlike boiling, keeps liquid just below its boiling point.",
    "A regulation ice hockey puck is 25.4 millimetres thick.",
    "A regular nonagon has nine equal sides and nine equal interior angles.",
    "The Elamite civilization predates the rise of the Persian Empire.",
    "A standard e-bike motor is typically rated between 250 and 750 watts.",
    "Orographic precipitation occurs when air is forced to rise over mountains.",
    "Julienning vegetables cuts them into thin, matchstick-shaped strips.",
    "A regulation curling stone weighs approximately 19 kilograms.",
    "A regular decagon has interior angles of 144 degrees.",
    "The Urartu kingdom was centered around Lake Van in eastern Anatolia.",
    "A standard action camera is often rated for underwater use to a set depth.",
    "Frontal precipitation develops along the boundary between air masses of different temperatures.",
    "Deboning a fish separates its flesh cleanly from the skeletal structure.",
    "A regulation biathlon rifle is fired from both standing and prone positions.",
    "A regular hendecagon has eleven equal sides.",
    "The Lydian kingdom is credited with minting some of the earliest coins.",
    "A standard fitness tracker measures steps using an accelerometer.",
    "Advection fog forms when warm, moist air moves over a cooler surface.",
    "Larding meat involves inserting strips of fat to add moisture during cooking.",
    "A regulation luge track descends with multiple banked curves.",
    "A regular icosagon has twenty equal sides and twenty equal angles.",
    "The Phrygian kingdom flourished in central Anatolia during the Iron Age.",
    "A standard GPS receiver calculates position using signals from multiple satellites.",
    "Radiation fog typically forms on clear nights with little wind.",
    "Spatchcocking a bird involves removing its backbone to flatten it for roasting.",
    "A regulation ski jump hill is measured by its calculated construction point.",
    "A rhombicuboctahedron has eight triangular faces and eighteen square faces.",
    "The Median Empire preceded the rise of the Achaemenid Persians.",
    "A standard barometer measures atmospheric pressure using mercury or an aneroid cell.",
    "Steam fog forms when cold air moves over comparatively warmer water.",
    "Confit involves slow-cooking meat submerged in its own rendered fat.",
    "A regulation speed skating oval is 400 metres per lap.",
    "A dodecahedron has twelve pentagonal faces.",
    "The Kingdom of Aksum minted its own coinage as early as the 3rd century.",
    "A standard anemometer measures wind speed using rotating cups or a vane.",
    "Upslope fog forms as moist air is lifted and cooled along rising terrain.",
    "Dredging food in flour before frying helps create a crisp exterior.",
    "A regulation figure skating rink is the same size as an ice hockey rink.",
    "An icosahedron has twenty equilateral triangular faces.",
    "The Kingdom of Nri influenced Igbo religious and political life for centuries.",
    "A standard hygrometer measures the relative humidity of the air.",
    "Valley fog forms when cold, dense air settles into low-lying terrain overnight.",
    "A regulation cross-country ski course varies in length by competition class.",
    "An octahedron has eight equilateral triangular faces.",
    "The Kingdom of Dahomey maintained an all-female military regiment.",
    "A standard altimeter estimates altitude from atmospheric pressure readings.",
    "Katabatic winds are common on the ice sheets of Antarctica and Greenland.",
    "Poaching an egg holds its shape using gently simmering, not boiling, water.",
    "A regulation modern pentathlon combines five distinct sporting disciplines.",
    "A tetrahedron has four triangular faces and four vertices.",
    "The Great Zimbabwe ruins were built without the use of mortar.",
    "A standard seismometer records ground motion during an earthquake.",
    "Föhn winds can cause rapid snowmelt on mountain slopes.",
    "Coddling an egg cooks it gently in a covered dish set in hot water.",
    "A regulation orienteering course uses a map and compass to navigate checkpoints.",
    "A hexagonal prism has two hexagonal bases and six rectangular sides.",
    "The Mali Empire's ruler Mansa Musa was famed for his wealth.",
    "A standard tide gauge measures the rise and fall of sea level.",
    "The Santa Ana winds are dry offshore winds common in Southern California.",
    "Coulis is a smooth sauce made by pureeing and straining fruit or vegetables.",
    "A standard rain gauge measures precipitation accumulated over a set period.",
    "A regulation sabre fencing bout is scored to fifteen touches.",
    "A truncated cone results from slicing the tip off a full cone.",
    "The Songhai Empire once controlled major trade routes across West Africa.",
    "A standard psychrometer uses wet and dry bulb thermometers to measure humidity.",
    "The mistral is a strong, cold wind that blows through southern France.",
    "Clarifying butter removes water and milk solids, leaving pure fat.",
    "A regulation épée fencing bout allows touches to the entire body.",
    "A frustum is the portion of a solid that remains after its top is cut off.",
    "The Kingdom of Kongo maintained diplomatic relations with Portugal from the 15th century.",
    "A standard ceilometer measures the height of cloud bases using a laser.",
    "The bora is a cold, gusty wind common along the Adriatic coast.",
    "Rendering lard involves melting pork fat slowly at low heat.",
    "A regulation foil fencing bout restricts valid touches to the torso.",
    "A rhombohedron has six rhombus-shaped faces.",
    "The Benin Empire was renowned for its intricate bronze plaques.",
    "A standard disdrometer measures the size and speed of falling raindrops.",
    "The sirocco carries warm, dust-laden air from the Sahara toward southern Europe.",
    "Proving dough a second time after shaping improves its final texture.",
    "A regulation modern épée weighs no more than 770 grams including its guard.",
]
N_TEXTS = len(ORIGINAL_CORPUS_201) + len(EXTENSION_CORPUS_299)
assert N_TEXTS == 500, f"expected 500 combined texts, got {N_TEXTS}"


def build_neutral_texts(smoke: bool = False) -> list[str]:
    if TEXTS_FILE.exists():
        return json.loads(TEXTS_FILE.read_text())["texts"]
    # Fixed order: original 201 first, extension 299 appended — this ordering
    # is itself deterministic and lets Stage B slice the first 201 to recover
    # the exact original corpus for the n=200-subsample bridge below.
    texts = list(ORIGINAL_CORPUS_201) + list(EXTENSION_CORPUS_299)
    if smoke:
        texts = texts[:40]
        return texts
    TEXTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEXTS_FILE.write_text(json.dumps({
        "source": "jamesrahenry/Rosetta_Analysis alignment/align_random_calib.py "
                   "RANDOM_CORPUS (201, recovered verbatim) + 299-sentence "
                   "extension in the same documented style, written 2026-07-17",
        "seed": SEED, "n": len(texts), "n_original": len(ORIGINAL_CORPUS_201),
        "n_extension": len(EXTENSION_CORPUS_299), "texts": texts,
    }))
    return texts


def _original_n200_subsample() -> list[str]:
    """Exact reproduction of align_random_calib.py's extract_random_activations:
    np.random.default_rng(seed=42).shuffle() on the 201-sentence original
    corpus, first 200 — the direct bridge to the published 0.1484 figure."""
    rng = np.random.default_rng(SEED)
    sentences = list(ORIGINAL_CORPUS_201)
    rng.shuffle(sentences)
    return sentences[:N_SUB]


# ---------------------------------------------------------------------------
# Stage A — per-model extraction
# ---------------------------------------------------------------------------


def extract_for_model(model_id: str, model, tok, device, batch_size: int,
                      upload_acts: bool = True, smoke: bool = False) -> None:
    from common import slugify
    slug = slugify(model_id)
    key = slug + ("_smoke" if smoke else "")
    if shard_done(JOB, key) is not None:
        log.info("[g5b] %s acts already extracted — skipping", slug)
        return

    offset = calibrate_offset(model, tok, device, slug, "causation", batch_size)
    texts = build_neutral_texts(smoke=smoke)
    t0 = time.time()
    acts = plain_acts(model, tok, texts, device, batch_size)  # [rows][n,d] f32
    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    npz = ACTS_DIR / f"{key}.npz"
    np.savez_compressed(npz, acts=np.stack(acts), offset=np.int64(offset))
    if upload_acts and not smoke:
        hf_upload(JOB, npz)
    shard_write(JOB, key, {
        "model_id": model_id, "n_texts": len(texts), "n_rows": len(acts),
        "offset": offset, "elapsed_s": time.time() - t0, "npz": npz.name,
    })
    log.info("[g5b] %s extracted %d rows in %.0fs", slug, len(acts), time.time() - t0)


# ---------------------------------------------------------------------------
# Stage B — pairwise Procrustes
# ---------------------------------------------------------------------------


def _load_acts(key: str) -> tuple[np.ndarray, int]:
    z = np.load(ACTS_DIR / f"{key}.npz")
    return z["acts"], int(z["offset"])


def _aligned_cos(src_cal: np.ndarray, tgt_cal: np.ndarray,
                 dom_src: np.ndarray, dom_tgt: np.ndarray) -> float:
    """Mirror rosetta_tools.alignment: R s.t. tgt_cal @ R ~= src_cal (both
    mean-centered, float64); aligned cosine = cos(dom_src, dom_tgt @ R)."""
    from scipy.linalg import orthogonal_procrustes, svd as _svd
    src_c = src_cal.astype(np.float64) - src_cal.mean(0, dtype=np.float64)
    tgt_c = tgt_cal.astype(np.float64) - tgt_cal.mean(0, dtype=np.float64)
    if not (np.isfinite(src_c).all() and np.isfinite(tgt_c).all()):
        raise ValueError("non-finite values in calibration acts — data problem, "
                         "not LAPACK flakiness; do not fall back")
    try:
        R, _ = orthogonal_procrustes(tgt_c, src_c)
    except np.linalg.LinAlgError:
        u, _, vt = _svd(tgt_c.T @ src_c, lapack_driver="gesvd")
        R = u @ vt
    v = dom_tgt @ R
    den = np.linalg.norm(dom_src) * np.linalg.norm(v)
    return float(np.dot(dom_src, v) / den) if den > 1e-12 else 0.0


def pairwise(concepts: list[str], smoke: bool = False,
             smoke_roster: list[str] | None = None) -> None:
    suffix = "_smoke" if smoke else ""
    if smoke:
        slugs = smoke_roster or []
    else:
        slugs = alignment_roster_from_hf()
        have = {p.stem for p in ACTS_DIR.glob("*.npz")}
        missing = [s for s in slugs if s not in have]
        if missing:
            log.warning("[g5b] %d roster models missing acts (reported, not fatal): %s",
                        len(missing), missing)
        slugs = [s for s in slugs if s in have]

    meta = {}
    for s in slugs:
        caz = load_caz(s, concepts[0])
        meta[s] = {"hidden_dim": caz["hidden_dim"], "family": family_of(s)}

    original_subsample = set(_original_n200_subsample())
    full_texts = build_neutral_texts(smoke=smoke)
    # Indices into the 500-text array (fixed order) corresponding to the
    # exact original n=200 subsample, for the byte-identical bridge.
    orig_idx = np.array([i for i, t in enumerate(full_texts) if t in original_subsample])
    assert smoke or len(orig_idx) == N_SUB, \
        f"expected {N_SUB} original-subsample texts located in the combined corpus, got {len(orig_idx)}"

    rows = []
    for a in slugs:
        for b in slugs:
            if a == b or meta[a]["hidden_dim"] != meta[b]["hidden_dim"]:
                continue
            if meta[a]["family"] == meta[b]["family"]:
                continue  # cross-family only, matching the published population
            acts_a, off_a = _load_acts(a + suffix)
            acts_b, off_b = _load_acts(b + suffix)
            for concept in concepts:
                caz_a, caz_b = load_caz(a, concept), load_caz(b, concept)
                la = peak_layer(caz_a) + off_a
                lb = peak_layer(caz_b) + off_b
                dom_a = dom_matrix(caz_a)[peak_layer(caz_a)]
                dom_b = dom_matrix(caz_b)[peak_layer(caz_b)]
                cal_a, cal_b = acts_a[la], acts_b[lb]
                n_full = min(len(cal_a), len(cal_b))
                row = {
                    "src": a, "tgt": b, "concept": concept,
                    "hidden_dim": meta[a]["hidden_dim"],
                    "aligned_cos_n500":
                        _aligned_cos(cal_a[:n_full], cal_b[:n_full], dom_a, dom_b),
                    "raw_cos": float(np.dot(dom_a, dom_b)
                                     / (np.linalg.norm(dom_a) * np.linalg.norm(dom_b))),
                }
                if not smoke:
                    row["aligned_cos_n200_original_subsample"] = _aligned_cos(
                        cal_a[orig_idx], cal_b[orig_idx], dom_a, dom_b)
                rows.append(row)
        log.info("[g5b] pairwise: %s done", a)

    c500 = [r["aligned_cos_n500"] for r in rows]
    c200o = [r["aligned_cos_n200_original_subsample"] for r in rows
             if "aligned_cos_n200_original_subsample" in r]
    out = {
        "job": JOB, "n_rows": len(rows), "n_models": len(slugs),
        "population": "ordered cross-family same-dimension pairs, alignment "
                      "roster (clusters A-E)",
        "corpus_note": "RECOVERED original 201-sentence corpus "
                       "(jamesrahenry/Rosetta_Analysis RANDOM_CORPUS, verbatim) "
                       "extended with 299 new sentences in the same documented "
                       "style to reach n=500. Supersedes g5's wikitext-103 "
                       "substitution — see module docstring.",
        "summary": {
            "grand_mean_n500": float(np.mean(c500)) if c500 else None,
            "grand_sd_n500": float(np.std(c500)) if c500 else None,
            "grand_mean_n200_original_subsample": float(np.mean(c200o)) if c200o else None,
            "published_n200_reference": 0.1484,
            "g5_wikitext_n500_reference": 0.2948,
            "g5_wikitext_n200_reference": 0.2291,
            "primary_reference": 0.9709,
        },
        "rows": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fname = f"g5b_random_text_null_original_corpus_n500{suffix}.json"
    fpath = OUT_ROOT / fname
    fpath.write_text(json.dumps(out, indent=1))
    if not smoke:
        hf_upload(JOB, fpath)
        hf_upload(JOB, TEXTS_FILE)
        hf_verify(JOB, [fname, TEXTS_FILE.name])
    log.info("[g5b] finalized: n500=%s  n200(original subsample)=%s  "
             "(published reference 0.1484)",
             out["summary"]["grand_mean_n500"],
             out["summary"]["grand_mean_n200_original_subsample"])


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="stage A for one HF model id")
    ap.add_argument("--extract-all", action="store_true",
                    help="stage A for the full alignment roster")
    ap.add_argument("--pairwise", action="store_true", help="stage B")
    ap.add_argument("--smoke", action="store_true",
                    help="pythia-160m + gpt2 (768-dim cross-family pair), 40 texts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-upload-acts", action="store_true")
    args = ap.parse_args()

    concepts = CONCEPTS_17[:2] if args.smoke else CONCEPTS_17
    smoke_models = ["EleutherAI/pythia-160m", "openai-community/gpt2"]

    if args.smoke:
        from common import slugify
        for mid in smoke_models:
            model, tok, device = load_model(mid)
            try:
                extract_for_model(mid, model, tok, device, args.batch_size,
                                  upload_acts=False, smoke=True)
            finally:
                release(model)
        pairwise(concepts, smoke=True,
                 smoke_roster=[slugify(m) for m in smoke_models])
        return

    if args.model:
        model, tok, device = load_model(args.model)
        try:
            extract_for_model(args.model, model, tok, device, args.batch_size,
                              upload_acts=not args.no_upload_acts)
        finally:
            release(model)

    if args.extract_all:
        roster = alignment_roster_from_hf()
        for slug in roster:
            mid = load_caz(slug, "causation")["model_id"]
            model, tok, device = load_model(mid)
            try:
                extract_for_model(mid, model, tok, device, args.batch_size,
                                  upload_acts=not args.no_upload_acts)
            finally:
                release(model)

    if args.pairwise:
        pairwise(concepts)


if __name__ == "__main__":
    main()
