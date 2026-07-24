#!/usr/bin/env python3
"""
Build data/heroes.json for the Glocal Teen Hero (Nepal) AT-SELECTION benchmark.

IMPORTANT METHODOLOGY: every person is scored on the record they had AT THE TIME
they were a Glocal honoree (the teen record the jury actually saw) -- NOT the
career they built in the years since. This is the only fair basis for the question
"would I be selected as a Glocal Teen Hero". A separate "now" note captures each
alum's later trajectory for a where-are-they-now view.

Scores 0-5 per dimension, hand-derived from deep public research per person.
Run:  python3 build.py
"""
import json, os
HERE=os.path.dirname(os.path.abspath(__file__))
RUBRIC={"description":"Seven dimensions from Glocal Teen Hero's stated criteria, scored on each person's AT-SELECTION teen record (not their later career). 0-5 each; weights sum to 1.0.",
 "dimensions":{
  "social_impact":{"weight":0.20,"definition":"Scale & measurability of positive change, at selection."},
  "leadership":{"weight":0.20,"definition":"Orgs founded/led, teams, mentoring, at selection."},
  "innovation":{"weight":0.15,"definition":"Technical depth/novelty of what they had built by then."},
  "entrepreneurship":{"weight":0.15,"definition":"Ventures/traction they had by then."},
  "recognition":{"weight":0.10,"definition":"Verifiable awards/media/credits held at selection."},
  "glocal_fit":{"weight":0.10,"definition":"Global-grade work rooted in local impact."},
  "character":{"weight":0.10,"definition":"Self-drive, resilience, initiative."}}}
DIMS=list(RUBRIC["dimensions"].keys())

# name | year | tier | si,ld,in,en,rc,gf,ch | conf | THEN (at-selection) | NOW (trajectory) | socials
DATA="""
Aarjan Chaudhary|2026|Applicant|4,5,5,5,5,5,5|high|Emergent Ventures fellow (the only one from Nepal), granted for Mecratus; CVE-2025-51588; research credited by Google, Twitch, EA & Stanford; founder of Arniko Hack Club (Nepal's largest teen tech community, 400+); ran Daydream (250+), Campfire (150+) & the InventionNovelty hacker house; building kroda.ai ($20k+ LOIs); security engineer at Dench (YC S24); exited ggamechamps (7k+ users)|Applicant (2026)|web=https://arjanchaudharyy.lol;x=https://x.com/arjanchaudharyy
Bipana Sharma|2015|Winner|5,4,1,1,4,4,4|med|Founded Ekta Child Club at 11; led anti-child-marriage/trafficking network; Asian Girls Human Rights Award 2015; helped make Sunwal a child-friendly municipality|Pursued LLB; child-rights advocacy|web=https://www.ted.com/talks/bipana_sharma_children_breaking_barriers
Santosh Lamichhane|2016|Winner|2,3,4,1,2,3,3|low|Self-taught teen inventor: corn-thresher machine, walking-robot car builds; won National Mechanical Exhibition|Untraceable|
Sachin Dangi|2017|Winner|3,4,2,3,3,3,3|low|President, Teenage Society of Nepal; co-founded Bizonomics & Skillathon|Youth organizer/writer; trail cold after 2019|
Prashansha KC|2018|Winner|4,4,2,1,3,4,4|med|21-day anti-kidnap-marriage campaign in Rukum (~2,400 reached); documentaries|Filmmaker; 'Iron Gate' at Sheffield DocFest 2022; UNICEF youth advocate|x=https://x.com/prashanshakc23
Samir Phuyal|2019|Winner|4,3,4,4,3,4,4|high|Built NayaKinmel + student apps; taught 10,000+ Django; 'Hamro Tech' YouTube|Founder/CEO Karobar (300k+ downloads); GSEA Nepal 2023; MIT Solve|li=https://linkedin.com/in/samirphuyal
Mandira Shrestha|2020|Winner|4,4,1,1,3,4,4|med|SRHR/child-rights; led child-authored UPR 2020 report; SRHR trainer|Registered Nurse; NGO boards|
Pranjal Chalise|2021|Winner|4,3,4,3,3,4,4|high|Drishti Nepal (open-source app for the blind); founded e-educators & Students Research Council Nepal|CS+Math at Amherst; Fermilab research|li=https://linkedin.com/in/pranjalchalise;gh=https://github.com/pranjalchalise
Rahul Ranjan Sah|2022|Winner|3,4,4,3,4,4,4|high|Bronze, Intl Astronomy Olympiad 2019; astronomy outreach to 7,000+; founded SXC|Furman grad; computational/quant research|li=https://linkedin.com/in/rahulranjansah;gh=https://github.com/rahulranjansah
Shruti Tiwari|2023|Winner|4,3,1,1,3,4,4|med|Plasticman Campaign + Chure tree-planting (WWF/Save the Children); child-marriage interventions|Trajectory unverified|
Ghanashyam Bishwakarma|2024|Winner|4,4,1,1,3,3,4|med|President, National Adolescent Boys Network Nepal; child-rights radio; World Social Forum 2024|Grassroots activism|
Krish Yadav|2025|Winner|3,4,3,3,4,3,4|med|Content creator; Associate Creative Head, The Nepali Comment; TNC Debates; Himalayan Linguistics Olympiad|Content creator (recent)|web=https://youtube.com/@TheNepaliComment
Aditya Khadka|2015|Finalist|3,2,4,1,4,4,3|med|Award-winning teen documentary 'Dhartiputra' (Chicago Children's FF etc.)|Unknown|
Avaneesh Yadav|2015|Finalist|2,2,2,2,1,2,2|low|Entrepreneur finalist; no footprint|Unknown|
Ravi Mandal|2015|Finalist|4,3,5,3,4,4,4|high|Microsoft Innovation Center apps; Imagine Cup Nepal 1st + world semifinalist; UNICEF/UNDP disaster apps|Founder/CEO ZeroTB (SF); Microsoft MVP|li=https://linkedin.com/in/ravimandal
Samprada Chapagain|2015|Finalist|2,2,1,1,1,2,2|low|Social-activist finalist; no footprint|Unknown|
Swastik Ghimire|2015|Finalist|3,3,1,1,2,3,3|low|Scouts/volunteering; writer|Unknown|
Anil Kharel|2016|Finalist|3,3,2,2,3,3,3|low|Top-6 finalist 2016 (age 15, Rupandehi)|Unknown|
Deepak B.K|2016|Finalist|2,2,1,1,1,2,2|low|Child-rights finalist; no footprint|Unknown|
Gaurav Pokhrel|2016|Finalist|2,2,1,1,2,3,3|low|Journalism finalist|Unknown|
Samata Shrestha|2016|Finalist|2,2,2,1,2,3,3|low|Poetry honoree (Jhapa)|Unknown|
Albina Prawin|2017|Finalist|5,4,2,2,4,5,5|med|Chair, Babiya Young Women Org (Plan Intl); Muslim girls' anti-child-marriage/GBV campaigner|Unknown|
Aayush Pandey|2017|Finalist|2,3,4,3,3,4,3|low|Teen programmer finalist (Rupandehi)|Unknown|
Prithu Singh Thakuri|2017|Finalist|2,3,3,4,3,4,3|high|Founded WPAll Club at 18; WordCamp Kathmandu speaker/organizer|Editorial Manager, RebelCode|li=https://linkedin.com/in/prithu-singh-thakuri
Rajaram Basnet|2017|Finalist|3,3,1,1,3,4,3|low|Domestic worker to President, District Child Club Network|Unknown|
Tanmay Chaudhary|2017|Finalist|2,3,3,2,3,4,3|med|Robotics Association of Nepal coordinator; built a 3-wheel RC car|UX designer/cinematographer|li=https://linkedin.com/in/tanmay-chaudhary-72173a175
Ashna Poudel|2017|20under20|4,4,2,3,3,5,4|med|Founder Sukarmi (skills training + paid employment for girls)|Business studies|
Bhabish Shrestha|2017|20under20|2,2,2,1,2,3,3|low|Social activist honoree|Unknown|
Pramish Paudel|2017|20under20|3,3,5,3,4,4,4|high|Built Proto News app (20k+ downloads) at 15; ICT Rising Star finalist|CV researcher; INSAIT intern|li=https://linkedin.com/in/pramish-paudel-554b2796;web=https://pramishp.github.io
Rizma Joshi|2017|20under20|2,2,2,1,2,2,3|low|Innovator honoree|Unknown|
Ruby Tamang|2017|20under20|1,2,1,1,2,2,3|low|Sports honoree|Unknown|
Pradip Adhikari|2017|20under20|1,2,2,1,2,2,3|low|Coder honoree|Unknown|
Prajesh Khanal|2017|20under20|4,4,3,2,4,5,4|med|Founder 'I Consume My Oxygen' (UNESCO ESD recognition)|UN MGCY; Youth Co:Lab|
Prashant Kandel|2017|20under20|1,2,2,1,2,2,3|low|App developer honoree|Unknown|
Bijay Acharya|2017|20under20|2,2,3,2,2,3,3|low|Innovator honoree|Unknown|
Biplov Jha|2017|20under20|2,3,4,2,3,4,3|med|Technology honoree|Ex-Tesla engineer; MS Texas A&M|li=https://linkedin.com/in/biplov-jha
Narayan Gautam|2017|20under20|3,3,1,1,2,3,3|low|Child activist honoree|Unknown|
Nivesh Kumar|2017|20under20|2,2,1,1,2,3,3|low|Journalist honoree|Unknown|
Sagar Parajuli|2017|20under20|4,3,2,2,3,4,4|med|Youth gender activist, UN Women Nepal|Public Health Officer|li=https://linkedin.com/in/sagar-parajuli-172732175
Sanjay Kumar Yadav|2017|20under20|2,2,1,1,2,3,3|low|Social activist honoree|Unknown|
Bikalpa Dhungana|2018|Finalist|3,3,5,2,4,3,3|med|Teen inventor: police-tested anti-drunk-driving cutoff, bomb-disposal robot; Yuba Pratibha Puraskar|EEE at Kathmandu University|li=https://linkedin.com/in/bikalpa-dhungana-b11b44197
Deepshikha Ghimire|2018|Finalist|3,4,2,2,3,4,3|med|Writer, Kathmandu Post bylines; founder AAYAM|Psychology; writer|
Saugat Tiwari|2018|Finalist|3,3,2,3,3,3,3|low|Ran youth 'Talk Series' (Chitwan)|Unknown|
Sudarshan Subedi|2018|Finalist|4,4,3,3,3,4,3|med|Co-founder Nepal Eco Club; British Council Schools Ambassador|British Council Nepal|li=https://linkedin.com/in/idebus95
Swornim Shrestha|2018|Finalist|2,3,3,4,3,3,3|med|Tinybits Foundation + Pahilo Deal e-commerce|E-commerce ads agency|
Aanchal Adhikari|2018|20under20|3,3,1,1,2,3,3|low|Social activist honoree|Unknown|
Aashutosh Sapkota|2018|20under20|2,3,3,3,2,3,3|low|Technology honoree|Unknown|
Abhishek Adhikari|2018|20under20|2,3,2,3,2,3,3|low|Entrepreneur honoree|Unknown|
Amit Khanal|2018|20under20|2,2,3,2,2,3,3|low|Technology honoree (age 15)|Unknown|
Anil Pradhan|2018|20under20|2,3,4,2,3,3,3|low|Built 'Smart Key' vehicle-security app; school science demos|Unknown|
Anisha Ruchal|2018|20under20|3,2,1,1,2,3,3|low|Social activist (age 14)|Unknown|
Deepa Adhikari|2018|20under20|3,2,2,2,2,3,3|low|Social activist honoree|Unknown|
Dipisha Bhujel|2018|20under20|3,3,1,1,2,3,3|med|Drug-addiction/child-rights awareness via flashmobs/drama (age 14)|Founder Sparsa (compostable pads); Iris STEM Prize; Zonta award|li=https://linkedin.com/in/dipisha-bhujel;web=https://theirisproject.org/winner/dipisha-bhujel
Kovid Raj Panthy|2018|20under20|2,3,4,3,3,4,3|med|IBM Champion 2019; coding author (age 14)|Founder Techsamaj; coding educator|li=https://linkedin.com/in/kovidpanthy
Nibesh Baral|2018|20under20|2,3,3,3,2,3,3|med|Social entrepreneur honoree|Creative Head, Ad Sathi|li=https://linkedin.com/in/nibeshnick
Palisha Shakya|2018|20under20|3,3,2,2,2,3,3|low|Drug-addiction/child-rights awareness (age 14)|Unknown|
Rhythm Sah|2018|20under20|1,2,3,2,2,3,3|low|Coder honoree|Unknown|
Sahil K Gupta|2018|20under20|2,2,4,2,3,4,3|low|Built Mars-rover model + drone; science expo (age 15)|Unknown|
Yatish Ojha|2018|20under20|3,3,2,2,3,3,3|med|Social activist honoree|Lawyer; Gen-Z Front figure; 2025 HoR candidate|
Bidhi Mandal|2019|Finalist|4,4,4,4,3,4,4|med|Won Infrastructure Idea Hunt; 2nd Yunus Challenge & UNDP Youth Co:Lab; plastic-to-brick venture|Compliance analyst, Fidelity|
Bikram Parajuli|2019|Finalist|3,4,4,3,3,4,3|med|Founded 'The Nepalions'; Karkhana maker-mentor; robotics programs|Software developer|
Lov Panthi|2019|Finalist|3,3,5,3,4,4,3|med|Co-built patent-registered first Nepali-speaking robot (with twin)|Lov Kush Robotics|
Rachin Kalakheti|2019|Finalist|2,3,4,3,3,4,3|med|Teen healthcare-AI + robotics work|Stanford; founder Cedro Finance|li=https://linkedin.com/in/akalakheti
Supriya Maharjan Sapkota|2019|Finalist|4,4,3,3,4,4,4|high|MARIAN Female Champion; Project Hope 2; won Infrastructure Idea Hunt|UX/branding, Dallas|x=https://x.com/supriyamss;web=https://supriyasapkota.com.np
Aanand Kumar Sahani|2019|20under20|3,2,1,1,3,3,3|low|Child-rights honoree (age 17)|Unknown|
Arjun Acharya|2019|20under20|3,2,2,1,3,3,3|low|Social activist/poet honoree|Unknown|
Babita Pariyar|2019|20under20|3,2,2,1,3,4,3|med|Child-marriage work; Ekikrit Child Club|Law student|
Bishnu Mijar|2019|20under20|4,4,2,2,3,5,4|med|Founder 'Ma Dalit?' anti-caste campaign|Dalit-rights activist|
Ganesh Sah Sudi|2019|20under20|4,3,4,2,4,5,4|med|Wildlife/snake rescuer; co-authored peer-reviewed krait paper (2020)|Mithila Wildlife Trust|web=https://mwt.org.np
Jyoti Singh|2019|20under20|3,2,3,3,2,3,3|low|Self-taught web dev; study apps (age 13)|Unknown|
Kovid Bhusan Pathak|2019|20under20|3,3,2,1,3,4,3|med|Fridays for Future Nepal; Sakhaa Nepal|Unknown|x=https://x.com/not_kovid
Nischal Bhandari|2019|20under20|3,2,1,1,2,3,3|low|Social activist honoree (age 18)|Unknown|
Rishi Kumar Gupta|2019|20under20|2,2,3,2,2,3,3|low|Innovator honoree|Unknown|
Rohan Bagale|2019|20under20|4,3,2,1,3,4,4|med|Child-rights advocacy|Child reintegration officer|
Samarth Jha|2019|20under20|3,3,2,1,2,3,3|low|'Mental mathematician'; school motivational programs|Unknown|
Sameer Chapagain|2019|20under20|2,3,2,1,3,3,3|low|Journalist honoree (Chitwan)|Unknown|
Shivu Pandey|2019|20under20|2,3,2,1,3,3,3|low|Public speaker; peer educator|US (Business IT/CS); recruiter|li=https://linkedin.com/in/shivu-pandey
Thalama Malla|2019|20under20|2,2,1,1,3,3,3|low|Social activist honoree (Nuwakot)|Unknown|
Ekraj Ghimire|2020|Finalist|3,3,3,3,4,4,3|med|Founded Butwal Robotics Club; Plant-for-the-Planet ambassador; campus app|IT professional|web=https://ekrajghimire.com.np
Reet Kafle|2020|Finalist|4,4,3,3,4,5,4|high|ETC Global Teacher Award 2020; early-childhood education work|Founder Early Years Stage Nepal|li=https://linkedin.com/in/reet-kafle-6739671a0;web=https://eysn.com.np
Subhash Sharma|2020|Finalist|2,2,3,2,3,4,3|low|Coder finalist (Janakpur)|Unknown|
Sulav Subedi|2020|Finalist|4,3,4,2,3,4,3|med|Self-taught roboticist; COVID assistive devices; mentored 30+|Unknown|
Vaibhav Nahata|2020|Finalist|4,4,3,4,4,5,4|high|Nepal Youth Icon; TEDx speaker; 'Being Champions' TV; Success Society International|Speaker/entrepreneur; Fulbright UGRAD|li=https://linkedin.com/in/championvaibhav;ig=https://instagram.com/championvaibhav
Abhishek Karna|2020|20under20|3,4,4,3,4,4,4|high|Founder 'Ecstatic Paradox' (physics+tech education)|Duke physics researcher|li=https://linkedin.com/in/abhishekkarna;web=https://abhishekkarna.com.np
Amit Timalsina|2020|20under20|3,4,3,3,3,4,4|high|Founder/President Young Scientists Community|AI entrepreneur (Blintic AI)|li=https://linkedin.com/in/amittimalsina
Ankit Mishra|2020|20under20|4,3,1,1,3,4,4|low|Child-rights (Nepalgunj)|Unknown|
Dikshya Gautam|2020|20under20|3,3,1,1,3,3,3|low|Child-rights; TV reporter|Unknown|
Grace Thapa|2020|20under20|3,3,3,4,3,4,4|med|Founder Graceful Nature (natural skincare)|Same brand|li=https://linkedin.com/in/grace-thapa-46843b1b3
Laxman Poudel|2020|20under20|3,4,5,3,4,5,4|high|Agrobot (MoEST-recognized); Anti-rape Watch (CAN best tech award)|ECE at Lafayette College|li=https://linkedin.com/in/coderlax;gh=https://github.com/techylax
Namrata Dahal|2020|20under20|4,4,2,2,3,5,4|med|VP Damak Child Network; stopped several child marriages|Unknown|
Preksha Dhami|2020|20under20|2,2,1,1,3,4,3|low|Social activist (Kailali)|Unknown|
Sanif Kandel|2020|20under20|4,4,3,2,3,5,4|med|Nepal Teen Leaders; 'We' for Change; climate organizing|Unknown|
Sanskriti Phuyal|2020|20under20|4,4,3,3,4,5,4|med|Founded 'Sports for Equality'; HER TURN Leadership Award|National cricketer (Bagmati)|web=https://cricnepal.com
Seliya Shrestha|2020|20under20|3,3,2,1,3,4,4|low|Child-rights RJ (Damak, age 15)|Unknown|
Sushant Sapkota|2020|20under20|5,4,3,3,4,5,4|high|Founder 'Go Green Go Clean' (300k+ reached), environmentalist|WAFF Global Teen Leader 2025; World Bank presenter|li=https://linkedin.com/in/susanleads
Swaraj Sagar Pradhan|2020|20under20|3,4,4,2,4,5,4|high|Gold, 8th APCYS 2019; SEDS 'Garuda' rocket; co-founder Physics Initiatives Nepal|Physics at Stony Brook|li=https://linkedin.com/in/swaraj-sagar-pradhan-827327139
Youbesh Dhaubhadel|2020|20under20|3,4,3,3,4,4,3|high|British Council 'Your World' national winner; photography|Econ at Idaho; US museum solo exhibition|li=https://linkedin.com/in/youbeshdhdl;ig=https://instagram.com/youbesh.dhdl;web=https://youbesh.com
Anurag Chapagain|2021|Finalist|3,3,3,3,3,4,3|med|Founder College Guide Nepal; science YouTube (~1,200 students)|University student|
Deepak Sutihar|2021|Finalist|3,3,2,2,3,4,3|med|Organized quiz raising ~Rs 3 lakh for an orphanage|Unknown|
Khusbu Bhandari|2021|Finalist|4,4,3,2,4,5,5|med|First girl snake-rescuer in Nepal (BBC News Nepal); Raise Hands Nepal|Healthcare in Canada|
Neha Gurung|2021|Finalist|4,4,3,2,4,4,4|med|First girl president, Ktm District Child Club Network; anti-online-abuse radio (10 FMs); 2 libraries|Unknown|
Sabhya Rai|2021|Finalist|4,4,3,3,4,5,4|high|Taught 500+ children in the pandemic; adviser at Sikaai; scholarship directory|CS at Fisk; Microsoft SWE intern|li=https://linkedin.com/in/sabhya-rai-420012228
Aabiskar Thapa Kshetri|2021|20under20|3,4,4,3,4,4,3|high|Founder Quantum Physics@SRCN; OneQuantum Nepal lead; ran 'Summer of Quantum'|CS at Lehigh; Goldman Sachs|li=https://linkedin.com/in/aabiskar-thapa-kshetri
Aaditya Singh Thapa|2021|20under20|3,3,1,1,3,4,3|low|Child-rights (Nepalgunj)|Unknown|
Amrit Rijal|2021|20under20|4,4,3,2,3,5,4|med|Founder '1000 hands, 500 trees' & Lakshyadeep|Unknown|
Anugraha Ghale|2021|20under20|4,4,3,4,4,5,4|high|Founder Gharmai Productions (art + mental-health social enterprise)|US Embassy Youth Council 2025|li=https://linkedin.com/in/anugrahaghale
Gobind Pajiyar|2021|20under20|3,3,3,4,3,4,3|high|Co-founder Griham & Pyume; coder's club; Dalit education work|Digital marketing officer|
Johnson Subedi|2021|20under20|4,4,4,4,3,4,4|med|Co-founder AuraED (digital literacy, 1000+ students)|Software developer|li=https://linkedin.com/in/johnson-subedi;gh=https://github.com/I-Johnson
Jwala Dhakal|2021|20under20|2,3,2,2,3,4,3|low|Young writer (Jhapa)|Unknown|
Mohan Budha|2021|20under20|3,3,2,2,3,4,3|med|Free radio education for marginalized students (Humla); YUWA|Unknown|
Om Prakash Wasti|2021|20under20|3,3,2,1,2,3,3|low|Youth activist (Kailali); radio presenter|Unknown|
Reyan Kumar Sapkota|2021|20under20|3,4,4,4,4,4,4|med|Founding board NECSA; Team Nepal captain IBCOL finals; LOCUS Young Innovator|Engineering at Pulchowk; EWB|li=https://linkedin.com/in/reyan-k-sapkota
Sabina Shakya|2021|20under20|4,3,4,3,3,4,4|med|Built 'Safa Sahar' recycling app; tree-planting; mentored 20+ girls|Conservation researcher (probable)|
Shubham Jha|2021|20under20|3,3,3,3,3,4,3|med|Directed ~24 short films; built Drishti Nepal & Kisan Nepal apps|IT/media|
Suraj Sapkota|2021|20under20|3,3,2,1,2,3,3|med|Taekwondo self-defense instructor; girls' workshops|Unknown|
Suyog Vardan Acharya|2021|20under20|2,2,3,2,3,4,3|med|Organic-chemistry sanitizer; drones/vacuum from waste (age 14)|Unknown|
Aashish Shah|2022|Finalist|3,4,4,4,3,4,3|med|Co-founded RoboTeach Nepal (25+ events, 20 schools, 2000+ students)|Same; Ashoka society|li=https://np.linkedin.com/in/aashishshah1
Bimarsha Poudel|2022|Finalist|3,3,3,3,3,4,3|med|Filmmaker; AYON short-film win; co-founder Seedlings Nepal|Filmmaker|
Darshana Rijal|2022|Finalist|5,4,3,3,4,5,4|high|Women Deliver Young Leader; VP YUWA; WHO temporary advisor|Same|web=https://yuwanepal.org
Nischal Singh Bista|2022|Finalist|4,3,3,4,4,4,3|med|Co-founder Bharyang Group (compostable products, ~28 lakh seed); ed projects 4,500+|Same|
Shrijana Gautam|2022|Finalist|4,4,3,2,4,4,4|med|VP 'We' for Change; led 'Hariyo Pusta' (1,500+ youth, 21 eco-clubs)|Unknown|
Aamod Paudel|2022|20under20|3,3,5,2,5,4,4|med|CERN Beamline for Schools shortlist; Imagine Cup Jr 1st; ESA Space Launchpad winner; Weizmann fellow|Computational science|web=https://aamodpaudel.com.np
Bidhata Pathak|2022|20under20|4,3,3,2,3,4,4|med|UNEP Tunza; CliMates Nepal; green-menstruation campaigns|Swarthmore (climate)|li=https://linkedin.com/in/bidhatapathak
Binita Dhakal|2022|20under20|4,4,3,4,3,4,4|high|Social entrepreneur; ex-Incubate Nepal researcher|CS at USM; Girls Who Code USM founder|li=https://linkedin.com/in/binitadhakal
Bishnu Shah|2022|20under20|4,3,2,3,3,4,4|low|Co-founder Revamp Youth Foundation; ENGin tutor|Unknown|
Ganga Sah|2022|20under20|4,3,2,1,3,4,4|med|Girls' education/anti-child-marriage (Mahottari); CWIN board|Unknown|web=https://school.digitalrightsnepal.org/fellow/ganga-sah
Kunal Sah|2022|20under20|3,4,3,4,3,3,3|high|Founder InternSathi (200+ placements)|Founder/CEO HireShore|li=https://linkedin.com/in/digitalkunalsah
Lucky Sah|2022|20under20|3,3,4,2,3,3,3|med|World Robotics Olympiad selection; NASA Space Apps|Unknown|
Prithak Shrestha|2022|20under20|3,2,1,1,2,3,3|low|Social activist honoree (Chitwan)|Unknown|
Rahul Mandal|2022|20under20|3,2,2,1,2,3,3|low|Tech educator honoree (Dhanusha)|Unknown|
Ranjan Shankar|2022|20under20|2,2,2,1,2,3,2|low|STEM education honoree (Terhathum)|Unknown|
Sampanna Jyoti Tuladhar|2022|20under20|4,4,3,3,3,4,4|high|Founded Beyond The Classroom (~10,000 students) with Nepal Economic Forum|Grinnell College|li=https://linkedin.com/in/sampannajtuladhar
Sambridhi Deo|2022|20under20|3,3,3,2,3,3,3|med|President, Initiative for Girls in Physics; polymer research|Artist/tech|
Sanskriti Duwadi|2022|20under20|2,2,2,1,2,3,2|low|Writer/gender activist honoree|Unknown|
Sarwagya Bhattarai|2022|20under20|3,3,2,2,3,3,3|med|Y-PEER peer educator (300+ SRHR)|Unknown|
Aryan Sigdel|2023|Finalist|3,3,3,4,3,4,3|med|Educational YouTube (1M+ views); jobs platform (1200+)|Content creator|
Atith Adhikari|2023|Finalist|4,4,4,4,3,4,3|med|Founded Sci-Pi (2020; ~200k visitors)|President, AskMattrab|li=https://linkedin.com/in/atith-adhikari;web=https://scipitutor.com
Madhav Khanal|2023|Finalist|3,4,3,2,5,4,4|high|Best-selling novel 'Avichal Karmayoddha' (10k+ copies); IPhO 2023 representative|CS at Rollins|li=https://linkedin.com/in/madhav-khanal-603b2a331
Nirajan Rimal|2023|Finalist|3,3,4,3,3,4,3|med|Innovation Exposure Series (robot battles, drone racing, rocket expo)|CEO, Astrobotech|
Preeti Pantha|2023|Finalist|3,3,3,2,3,4,3|med|Founder, The Orbona kids' magazine|Student; Incubate Nepal|li=https://linkedin.com/in/preeti-pantha-132a98279
Avinash Kumar Paswan|2023|20under20|3,3,4,3,3,4,3|med|Pioneered 'MAITH HOP' Maithili fusion music|Producer|
Dikshya Bharati|2023|20under20|3,3,2,1,3,3,3|low|EU Youth Sounding Board; digital literacy|Unknown|
Diwash Sarraf|2023|20under20|4,3,3,2,3,4,3|med|UNICEF/CWIN mental-health advocacy; chatbot contribution|Unknown|
Prakash Badu|2023|20under20|3,3,2,1,3,4,3|med|UNFPA Rupantaran facilitator; anti-child-marriage (age 15)|Unknown|
Prakash Pant|2023|20under20|2,3,4,1,4,4,4|high|Represented Nepal at IMO 2023 (2nd-highest Nepali score)|Math at UVM|li=https://linkedin.com/in/prakash-pant-7786711ab
Raushan Pandit|2023|20under20|2,2,2,1,2,3,2|low|Tech enthusiast honoree|Unknown|
Risham Kumar Sah|2023|20under20|3,3,3,2,3,4,3|med|Mentored Nepal's first Rocket Camp; 'Bambana' eco-packaging (200+ farmers)|Unknown|
Sadiksha Ghimire|2023|20under20|3,2,1,1,3,4,3|low|Health activist honoree|Unknown|
Sagar Budha|2023|20under20|4,3,2,2,3,4,3|med|'Clean and Green Surkhet' (30,000+ trees; 100+ clean-ups)|Unknown|
Sagar Gupta|2023|20under20|2,2,3,1,3,3,3|low|AI researcher; cardiovascular math model|Unknown|
Samip Paudel|2023|20under20|2,2,3,1,3,3,3|low|AI/ML enthusiast (age 16)|Unknown|
Shubham Upreti|2023|20under20|3,3,2,3,3,4,3|med|Founder Student Alliance For Creation (art/heritage)|Unknown|
Sumitra Acharya|2023|20under20|3,2,1,1,3,4,3|med|Child-rights (Ekikrit Child Club); Plasticman|Unknown|
Sushan Shrestha|2023|20under20|3,3,1,1,3,4,3|med|President, Municipal Child Club Network (Sunwal)|Unknown|
Aashish Panthi|2024|Finalist|4,4,4,3,4,5,4|high|Cosog Nepal 'Code for Charity' (3,000+ students); Surakshya app; Koded YouTube|Same|li=https://linkedin.com/in/aashishpanthi;gh=https://github.com/aashishpanthi;web=https://aashishpanthi.name.np
Nischal Bhattarai|2024|Finalist|4,4,3,2,4,5,4|med|Public-speaking programs (1,235+ teens); cybercrime-safety education|Student|
Purnima Timsina|2024|Finalist|4,4,2,1,4,4,4|med|President DPL Teens; Board, National Adolescent Girls' Network; child-protection|Paramedic|
Sajani Sharma|2024|Finalist|4,3,2,1,4,4,3|med|Women's empowerment; clean-ups; tree-planting|Unknown|
Shreejay Subedi|2024|Finalist|4,4,4,4,4,5,4|high|Founder Venture Tech Nepal (bus app 1000+ users); award-winning documentary; Nayachapter 300+|CS at Howard; AI-bias research|li=https://linkedin.com/in/shreejay-subedi-168735213
Aadesh Regmi|2024|20under20|3,3,2,1,3,4,3|low|Social activist/education honoree (Parbat)|Unknown|
Aayushman Puri|2024|20under20|4,3,2,1,3,4,4|med|YUWA Youth Council; SRHR/CSE peer advocacy|Public health student|
Aryan Basnet|2024|20under20|4,4,4,3,3,5,4|med|Authored 'Foundations of ML for High Schoolers'; AI-for-TB research; Nepal Hiking Society co-founder|IT/AI educator|
Ashish Banjara|2024|20under20|4,3,2,1,3,4,4|med|Radio education debate; 'Reusing the Book' (3,000+ students)|Unknown|
Hangsam Nembang|2024|20under20|3,4,3,4,4,5,3|med|Founder Junior Entrepreneurship Circle (1,500+); TEDx event head; Jiva Organics|KMC full-ride scholar|li=https://linkedin.com/in/hangsam-nembang-bb3520238
Kaushal Niraula|2024|20under20|4,4,3,2,4,5,4|med|Founder Climate Care Network; British Council Youth for Climate|Env science at KU|ig=https://instagram.com/niraulakaushal
Kishor Shahi|2024|20under20|3,2,2,1,3,4,3|low|Climate (Dailekh); surveyed 100+ farmers|Unknown|
Krishtina Khanal|2024|20under20|4,3,4,3,4,5,3|med|Cosog outreach; designer, Ragat Nepal blood platform (5,000+ donors); DevTrack|Tech-for-good|
Prashim Timsina|2024|20under20|3,4,4,4,4,4,4|high|Built BloodDonor, Khoja, DevTrack; NASA Cubes-in-Space project|Software engineer; Incubate Nepal|li=https://linkedin.com/in/prashimpy;gh=https://github.com/prashim;web=https://prashim.com.np
Purnima Timsina 2|2024|skip|0,0,0,0,0,0,0|low|dup|dup|
Saurab Banstola|2024|20under20|4,4,4,3,4,4,4|high|Founder Rising Pupil (12,000+ students); co-authored calculus book (with Univ. of Sydney prof)|Teacher, Bloom Nepal|li=https://linkedin.com/in/saurab-banstola-b3ab561b0
Shakti K.C.|2024|20under20|3,3,3,3,3,3,3|med|IJSO 2021 Nepal finalist; asteroid-hunting (IASC)|Frontend/AI dev|web=https://kcshakti.com.np
Sinshiya K.C.|2024|20under20|4,4,2,2,4,4,4|med|Menstrual hygiene/CSE sessions across all municipality wards|Unknown|
Sugam Parajuli|2024|20under20|3,3,3,4,4,4,3|med|MD, Sugam Computer Sewa; Incubate Nepal researcher (DevTrack)|Unknown|
Tushar Shah|2024|20under20|4,3,4,4,4,4,3|med|Work Root Venture (60+ businesses); DevTrack|Unknown|
Manushi Neupane|2025|Finalist|4,4,3,3,4,4,4|high|Co-builder Pyari Periods (Kathmandu Post); Smart Cheli; Engineers Without Borders Nepal|Yale (full ride)|li=https://linkedin.com/in/manushi-neupane-6080bb2bb;web=https://pyari.org
Nischal Dhungana|2025|Finalist|3,3,2,1,3,4,4|med|Represented Climate Vulnerable Forum / V20 at climate dialogues|Env science student|
Oxford Acharya|2025|Finalist|3,4,2,2,3,5,4|med|Founder United Junior Red Cross Circle (2,000-student network claim); age 15, rural Jumla|Student|web=https://thegcsc.org
Renuka Singh|2025|Finalist|3,3,3,3,3,4,3|med|Rooftop/organic dragon-fruit farming; Women LEAD grad|Agro-entrepreneur|
Safal Poudel|2025|Finalist|3,3,4,3,3,4,3|high|Builds AI tools (PlantMD, Medforce AI, AgroBot); NPLCoder; shipped GitHub code|Web/AI dev|li=https://linkedin.com/in/safal-poudel-0900581b8;gh=https://github.com/safal808;web=https://safal-poudel.com.np
Aawish Khanal|2025|20under20|2,2,2,3,2,3,3|low|Aspiring entrepreneur (Butwal)|Unknown|
Dhurbesh Dhami|2025|20under20|3,3,2,1,2,4,3|med|Public-health/child-rights campaigns; Leo Club (Bajura)|Unknown|
Gokul Shrestha|2025|20under20|4,3,4,2,4,4,4|high|2024 Rise Global Winner (elite, Rhodes/Schmidt); corn-based water filtration; APCYS bronze|Researcher|web=https://risefortheworld.org/winners/gokul-shrestha
Nishant Raj Sarraf|2025|20under20|2,3,3,3,2,3,3|med|Founder RedPaper (legal-tech); Polygence scholar; iGEM remote fellow|AI/social innovator|web=https://polygence.org/scholars/nishant-raj-sarraf
Mohammad Aftab Sheikh|2025|20under20|2,3,3,4,2,3,3|med|Co-founder Grocery Gunj & Build Nepal Group; rapper|Entrepreneur|li=https://linkedin.com/in/mohammadaftabsheikh
Osish Niraula|2025|20under20|3,3,2,2,3,4,3|med|IOAA 2024 (selective); co-founder Learn2Lead Nepal; UN HLPF 2023|Unknown|li=https://linkedin.com/in/osish-niraula-ab7202282
Phurwa Tsering Gurung|2025|20under20|2,2,3,2,2,4,3|low|Documentary filmmaker (Dolpo)|Unknown|
Pooja Mainali|2025|20under20|2,3,1,1,2,3,3|low|Youth-leadership coordination|Unknown|
Rakshit Poudel|2025|20under20|3,3,2,2,2,4,3|med|Physics/STEM educator; Olympiad math teaching|Unknown|li=https://linkedin.com/in/rakshit-poudel-659230374
Ritu Gharti|2025|20under20|3,2,1,1,2,4,3|low|Peer educator (Nawalpur)|Unknown|
Ruchi Ojha|2025|20under20|3,3,2,2,2,3,3|med|'Katha Yatra' menstrual-health; certified MHM trainer|Unknown|li=https://linkedin.com/in/ruchi-ojha-9ba2a2283
Sajan Adhikari|2025|20under20|3,3,2,2,2,3,3|med|Founder Initiation Lakshya (cybersecurity/sustainability); TEDx organizer|Unknown|li=https://linkedin.com/in/sajan-adhikari-10620620a
Saksham Ghimire|2025|20under20|3,3,3,3,2,4,3|high|Founder Hamro Niti (civic/policy newsletter with real published articles)|Same|li=https://linkedin.com/in/sakshamghimire10;web=https://hamroniti.com
Saksham Rupakheti|2025|20under20|3,3,3,3,2,4,3|med|Co-founder ThinkNiti Foundation (STEM access); TEDxBaneshwor|Unknown|
"""

def parse_socials(s):
    d={}
    for kv in s.split(";"):
        kv=kv.strip()
        if "=" in kv:
            k,v=kv.split("=",1); d[k.strip()]=v.strip()
    return d

heroes=[]
for line in DATA.strip().splitlines():
    p=[x.strip() for x in line.split("|")]
    if len(p)<8: continue
    name,year,tier,scores,conf,then,now,socials=p[0],int(p[1]),p[2],p[3],p[4],p[5],p[6],p[7]
    if tier=="skip": continue
    sv=[int(x) for x in scores.split(",")]
    s={DIMS[i]:sv[i] for i in range(7)}
    heroes.append({"name":name,"year":year,"award":tier,"s":s,"conf":conf,
                   "then":then,"now":now,"links":parse_socials(socials),
                   "me":tier=="Applicant","est":conf=="low"})

out={"rubric":RUBRIC,
 "note":"AT-SELECTION benchmark: each person scored on the record they held when they were a Glocal honoree, NOT their later career. 'now' field holds their subsequent trajectory for a separate view.",
 "heroes":heroes}
json.dump(out,open(os.path.join(HERE,"data","heroes.json"),"w"),indent=1)
open(os.path.join(HERE,"corpus.js"),"w").write("window.__CORPUS__="+json.dumps(heroes,separators=(',',':'))+";")

W={k:v["weight"] for k,v in RUBRIC["dimensions"].items()}
tot=lambda s:sum(s[k]*W[k] for k in W)
wins=[h for h in heroes if h["award"]=="Winner"]
print(f"heroes:{len(heroes)} winners:{len(wins)} finalists:{sum(1 for h in heroes if h['award']=='Finalist')} 20u20:{sum(1 for h in heroes if h['award']=='20under20')}")
rank=sorted(heroes,key=lambda h:-tot(h["s"]))
print("TOP 12 (at-selection):")
for i,h in enumerate(rank[:12],1): print(f"{i:>2}. {h['name']:<24}{h['year']} {h['award']:<10}{tot(h['s']):.2f}")
