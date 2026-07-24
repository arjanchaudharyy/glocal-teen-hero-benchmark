#!/usr/bin/env python3
"""
Build data/heroes.json from the full Glocal Teen Hero roster.

Two scoring tiers, both documented and reproducible:
  * WINNERS + a few deeply-profiled honorees  -> hand-scored from public records (SCORES below).
  * The wider cohort (20under20 + finalists)   -> a transparent tier+field HEURISTIC.
    These are labeled est=True so the UI/README can flag them as modeled, not researched.

Run:  python3 build.py   ->  writes data/heroes.json  (+ corpus.js for the web app)
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

RUBRIC = {
  "description": "Dimensions from Glocal Teen Hero's stated criteria + organizer write-ups of why winners won. 0-5 each, weights sum to 1.0.",
  "dimensions": {
    "social_impact":   {"weight":0.20,"definition":"Scale and measurability of positive change for people/community."},
    "leadership":      {"weight":0.20,"definition":"Founding/leading orgs, teams, movements; mentoring others."},
    "innovation":      {"weight":0.15,"definition":"Technical depth and novelty of what was actually built."},
    "entrepreneurship":{"weight":0.15,"definition":"Ventures, traction, revenue/commitments, exits."},
    "recognition":     {"weight":0.10,"definition":"External verifiable credibility: media, awards, credits, CVEs."},
    "glocal_fit":      {"weight":0.10,"definition":"Global-grade work rooted in local (Nepal/South Asia) impact."},
    "character":       {"weight":0.10,"definition":"Self-drive, resilience, initiative, will to learn."},
  }
}
DIMS = list(RUBRIC["dimensions"].keys())

# ---- hand-scored: winners + Arjan + one enriched honoree (from detailed public records) ----
HAND = {
 ("Bipana Sharma",2015):        (dict(social_impact=5,leadership=4,innovation=2,entrepreneurship=2,recognition=5,glocal_fit=4,character=5),
   "Founded Ekta Child Club at 11; campaigned against child marriage/trafficking. Asian Girls Human Rights Award; her story became the UNICEF/Govt film 'JYOTI'; TED speaker.",
   {"web":"https://www.ted.com/talks/bipana_sharma_children_breaking_barriers"}),
 ("Santosh Lamichhane",2016):   (dict(social_impact=2,leadership=3,innovation=5,entrepreneurship=3,recognition=3,glocal_fit=4,character=5),
   "Self-taught mechanical inventor: corn-thresher machine, car-to-walking-robot builds; won the National Mechanical Exhibition; science advisor on a feature film.",
   {"web":"https://instagram.com/lamichhane_santosh"}),
 ("Sachin Dangi",2017):         (dict(social_impact=4,leadership=5,innovation=2,entrepreneurship=4,recognition=3,glocal_fit=3,character=4),
   "President of the Teenage Society of Nepal (~5,000 members); co-founded Bizonomics and Skillathon; NY Academy of Sciences ambassador.",{}),
 ("Prashansha KC",2018):        (dict(social_impact=5,leadership=4,innovation=2,entrepreneurship=2,recognition=5,glocal_fit=4,character=5),
   "Ran the 21-day 'Eradication of Kidnap Marriage' project in Rukum (~2,400 reached); Zonta Young Women in Public Affairs Award; UNICEF Nepal youth advocate; TED speaker.",
   {"x":"https://x.com/prashanshakc23"}),
 ("Samir Phuyal",2019):         (dict(social_impact=4,leadership=3,innovation=4,entrepreneurship=4,recognition=3,glocal_fit=4,character=4),
   "Built NayaKinmel.com and student apps; taught 10,000+ people Django; runs the 'Hamro Tech' YouTube channel teaching web dev in Nepali.",{}),
 ("Mandira Shrestha",2020):     (dict(social_impact=5,leadership=4,innovation=2,entrepreneurship=2,recognition=4,glocal_fit=3,character=5),
   "Health/SRHR activist and RN; led the child-led UPR 2020 report; founder of 'Triple H'; Asia-Pacific Youth of the Year 2024.",{}),
 ("Pranjal Chalise",2021):      (dict(social_impact=4,leadership=3,innovation=4,entrepreneurship=3,recognition=3,glocal_fit=4,character=4),
   "Built Drishti Nepal (open-source app for the visually impaired); founded e-educators Nepal and Students Research Council Nepal.",
   {"li":"https://linkedin.com/in/pranjalchalise/","web":"https://github.com/pranjalchalise"}),
 ("Rahul Ranjan Sah",2022):     (dict(social_impact=3,leadership=4,innovation=5,entrepreneurship=4,recognition=4,glocal_fit=4,character=5),
   "Astropreneur behind 'Arohan' (toward Nepal's first private space mission); astronomy outreach to 7,000+ rural students; IAAC ambassador; now at Furman University.",
   {"li":"https://linkedin.com/in/rahulranjansah/","web":"https://rahulranjansah.com.np"}),
 ("Shruti Tiwari",2023):        (dict(social_impact=5,leadership=4,innovation=2,entrepreneurship=2,recognition=5,glocal_fit=4,character=5),
   "Plasticman Campaign + Chure tree-planting (WWF/Save the Children); helped stop 12 child marriages; mentored ~20 Dalit/Janjati girls; interviewed by BBC UK.",{}),
 ("Ghanashyam Bishwakarma",2024):(dict(social_impact=5,leadership=5,innovation=2,entrepreneurship=2,recognition=4,glocal_fit=3,character=5),
   "8+ yrs activism; President of the National Adolescent Boys Network Nepal; radio host on child rights; represented Nepal at World Social Forum 2024.",{}),
 ("Krish Yadav",2025):          (dict(social_impact=3,leadership=4,innovation=3,entrepreneurship=4,recognition=5,glocal_fit=3,character=4),
   "Content creator; Associate Creative Head at The Nepali Comment (405k+ subs); hosts TNC Debates (15.5k+); leads the Himalayan Linguistics Olympiad and youth-ed projects.",
   {"web":"https://youtube.com/@TheNepaliComment"}),
 ("Nishant Raj Sarraf",2025):   (dict(social_impact=4,leadership=4,innovation=5,entrepreneurship=4,recognition=4,glocal_fit=4,character=4),
   "AI researcher; founder of RedPaper (youth legal-tech); Stanford iGEM remote research fellow building AI for early autism detection via speech.",
   {"web":"https://polygence.org/scholars/nishant-raj-sarraf"}),
 ("Aarjan Chaudhary",2026):     (dict(social_impact=4,leadership=5,innovation=5,entrepreneurship=5,recognition=5,glocal_fit=5,character=5),
   "'Youngest Hacker of Nepal'; CVE-2025-51588; credited by Google, Twitch, EA, Stanford. Security engineer at Dench (YC S24); founder of kroda.ai ($20k+ LOIs) and Arniko Hack Club (400+ members, top-10 teen non-profit in Asia); ran Daydream (250+) and Campfire (150+); 1x exit.",
   {"web":"https://arjanchaudharyy.lol","x":"https://x.com/arjanchaudharyy"}),
}

WINNER_KEYS = set(HAND) - {("Nishant Raj Sarraf",2025),("Aarjan Chaudhary",2026)}

# ---- roster of the wider cohort: "name | year | edition | tier | field | home" ----
# tier: finalist (top-6, non-winner) or 20under20. Winners live in HAND above.
ROSTER = """
# 2015-2016 finalists
Aditya Khadka|2015|Nepal|finalist|documentary|;Avaneesh Yadav|2015|Nepal|finalist|entrepreneur|;Ravi Mandal|2015|Nepal|finalist|app developer|;Samprada Chapagain|2015|Nepal|finalist|social activist|;Swastik Ghimire|2015|Nepal|finalist|writer|
Anil Kharel|2016|Nepal|finalist|social|;Deepak B.K|2016|Nepal|finalist|child rights|;Gaurav Pokhrel|2016|Nepal|finalist|journalism|;Samata Shrestha|2016|Nepal|finalist|poetry|
# 2017 20under20
Albina Prawin|2017|Nepal|20under20|social activist|Inaruwa;Aayush Pandey|2017|Nepal|20under20|coder|;Ashna Poudel|2017|Nepal|20under20|social activist|;Bhabish Shrestha|2017|Nepal|20under20|social activist|;Pramish Paudel|2017|Nepal|20under20|technology|;Rajaram Basnet|2017|Nepal|20under20|child activist|;Rizma Joshi|2017|Nepal|20under20|innovator|;Ruby Tamang|2017|Nepal|20under20|sports|;Pradip Adhikari|2017|Nepal|20under20|coder|;Prajesh Khanal|2017|Nepal|20under20|environmentalist|;Prashant Kandel|2017|Nepal|20under20|app developer|;Prithu Singh Thakuri|2017|Nepal|20under20|entrepreneur|;Bijay Acharya|2017|Nepal|20under20|innovator|;Biplov Jha|2017|Nepal|20under20|technology|;Narayan Gautam|2017|Nepal|20under20|child activist|;Nivesh Kumar|2017|Nepal|20under20|journalist|;Sagar Parajuli|2017|Nepal|20under20|social activist|;Sanjay Kumar Yadav|2017|Nepal|20under20|social activist|;Tanmay Chaudhary|2017|Nepal|20under20|innovator|
# 2018 20under20
Aanchal Adhikari|2018|Nepal|20under20|social activist|;Aashutosh Sapkota|2018|Nepal|20under20|technology|;Abhishek Adhikari|2018|Nepal|20under20|entrepreneur|;Amit Khanal|2018|Nepal|20under20|technology|;Anil Pradhan|2018|Nepal|20under20|technology|;Anisha Ruchal|2018|Nepal|20under20|social activist|;Bikalpa Dhungana|2018|Nepal|20under20|innovator|;Deepa Adhikari|2018|Nepal|20under20|social activist|;Deepshikha Ghimire|2018|Nepal|20under20|social activist|;Dipisha Bhujel|2018|Nepal|20under20|social activist|;Kovid Raj Panthy|2018|Nepal|20under20|coder|;Nibesh Baral|2018|Nepal|20under20|social activist|;Palisha Shakya|2018|Nepal|20under20|social activist|;Rhythm Sah|2018|Nepal|20under20|coder|;Sahil K Gupta|2018|Nepal|20under20|technology|;Saugat Tiwari|2018|Nepal|20under20|entrepreneur|;Sudarshan Subedi|2018|Nepal|20under20|ecopreneur|;Swornim Shrestha|2018|Nepal|20under20|entrepreneur|;Yatish Ojha|2018|Nepal|20under20|social activist|
# 2019 20under20
Aanand Kumar Sahani|2019|Nepal|20under20|child rights|;Arjun Acharya|2019|Nepal|20under20|social activist|;Babita Pariyar|2019|Nepal|20under20|child rights|;Bidhi Mandal|2019|Nepal|20under20|entrepreneur|;Bikram Parajuli|2019|Nepal|20under20|tech educator|;Bishnu Mijar|2019|Nepal|20under20|social activist|;Ganesh Sah Sudi|2019|Nepal|20under20|wildlife conservation|;Jyoti Singh|2019|Nepal|20under20|programmer|;Kovid Bhusan Pathak|2019|Nepal|20under20|social activist|;Lov Panthi|2019|Nepal|20under20|innovator|;Nischal Bhandari|2019|Nepal|20under20|social activist|;Rachin Kalakheti|2019|Nepal|20under20|technopreneur|;Rishi Kumar Gupta|2019|Nepal|20under20|innovator|;Rohan Bagale|2019|Nepal|20under20|social activist|;Samarth Jha|2019|Nepal|20under20|social activist|;Sameer Chapagain|2019|Nepal|20under20|journalist|;Shivu Pandey|2019|Nepal|20under20|cyber security|;Supriya Maharjan Sapkota|2019|Nepal|20under20|social activist|;Thalama Malla|2019|Nepal|20under20|social activist|
# 2020 20under20
Abhishek Karna|2020|Nepal|20under20|technopreneur|Mahottari;Amit Timalsina|2020|Nepal|20under20|technopreneur|Rupandehi;Ankit Mishra|2020|Nepal|20under20|child rights|Banke;Dikshya Gautam|2020|Nepal|20under20|child rights|Kathmandu;Ekraj Ghimire|2020|Nepal|20under20|environmentalist|Palpa;Grace Thapa|2020|Nepal|20under20|entrepreneur|Lalitpur;Laxman Poudel|2020|Nepal|20under20|innovator|Rupandehi;Namrata Dahal|2020|Nepal|20under20|child rights|Jhapa;Preksha Dhami|2020|Nepal|20under20|social activist|Kailali;Reet Kafle|2020|Nepal|20under20|educator|Morang;Sanif Kandel|2020|Nepal|20under20|youth activist|Rupandehi;Sanskriti Phuyal|2020|Nepal|20under20|social activist|Kathmandu;Seliya Shrestha|2020|Nepal|20under20|social activist|Jhapa;Subhash Sharma|2020|Nepal|20under20|coder|Janakpur;Sulav Subedi|2020|Nepal|20under20|innovator|Ilam;Sushant Sapkota|2020|Nepal|20under20|environmentalist|Surkhet;Swaraj Sagar Pradhan|2020|Nepal|20under20|programmer|Lalitpur;Vaibhav Nahata|2020|Nepal|20under20|entrepreneur|Morang;Youbesh Dhaubhadel|2020|Nepal|20under20|photographer|Kathmandu
# 2021 20under20 (+finalists)
Aabiskar Thapa Kshetri|2021|Nepal|20under20|engineering|Pyuthan;Aaditya Singh Thapa|2021|Nepal|20under20|child rights|Banke;Amrit Rijal|2021|Nepal|20under20|child rights|Sunsari;Anugraha Ghale|2021|Nepal|20under20|entrepreneur|Lalitpur;Anurag Chapagain|2021|Nepal|finalist|content creator|Nawalparasi;Deepak Sutihar|2021|Nepal|finalist|social entrepreneur|Saptari;Gobind Pajiyar|2021|Nepal|20under20|entrepreneur|Siraha;Johnson Subedi|2021|Nepal|20under20|technopreneur|Parbat;Jwala Dhakal|2021|Nepal|20under20|writer|Jhapa;Khusbu Bhandari|2021|Nepal|finalist|wildlife conservation|Chitwan;Mohan Budha|2021|Nepal|20under20|social activist|Humla;Neha Gurung|2021|Nepal|finalist|social activist|Kathmandu;Om Prakash Wasti|2021|Nepal|20under20|social activist|Kailali;Reyan Kumar Sapkota|2021|Nepal|20under20|social entrepreneur|Bhaktapur;Sabhya Rai|2021|Nepal|finalist|education activist|Ilam;Sabina Shakya|2021|Nepal|20under20|environmentalist|Lalitpur;Shubham Jha|2021|Nepal|20under20|filmmaker|Mahottari;Suraj Sapkota|2021|Nepal|20under20|social activist|Nawalparasi;Suyog Vardan Acharya|2021|Nepal|20under20|innovator|Kaski
# 2022 20under20 (+finalists)
Aamod Paudel|2022|Nepal|20under20|tech activist|Rupandehi;Aashish Shah|2022|Nepal|finalist|edu-technologist|Dhanusha;Bidhata Pathak|2022|Nepal|20under20|activist|Nuwakot;Bimarsha Poudel|2022|Nepal|finalist|filmmaker|Chitwan;Binita Dhakal|2022|Nepal|20under20|social entrepreneur|Gorkha;Bishnu Shah|2022|Nepal|20under20|education activist|Dhanusha;Darshana Rijal|2022|Nepal|finalist|women rights|Morang;Ganga Sah|2022|Nepal|20under20|female education|Mahottari;Kunal Sah|2022|Nepal|20under20|tech-intrapreneur|Morang;Lucky Sah|2022|Nepal|20under20|technology|Mahottari;Nischal Singh Bista|2022|Nepal|finalist|ecopreneur|Achham;Prithak Shrestha|2022|Nepal|20under20|social activist|Chitwan;Rahul Mandal|2022|Nepal|20under20|tech educator|Dhanusha;Ranjan Shankar|2022|Nepal|20under20|STEM education|Terhathum;Sampanna Jyoti Tuladhar|2022|Nepal|20under20|education|Kathmandu;Sambridhi Deo|2022|Nepal|20under20|artist|Hetauda;Sanskriti Duwadi|2022|Nepal|20under20|gender activist|Kathmandu;Sarwagya Bhattarai|2022|Nepal|20under20|health activist|Kathmandu;Shrijana Gautam|2022|Nepal|finalist|social activist|Morang
# 2023 20under20 (+finalists)
Aryan Sigdel|2023|Nepal|finalist|educator|;Atith Adhikari|2023|Nepal|finalist|social entrepreneur|;Avinash Kumar Paswan|2023|Nepal|20under20|musician|;Dikshya Bharati|2023|Nepal|20under20|health activist|;Diwash Sarraf|2023|Nepal|20under20|social activist|;Madhav Khanal|2023|Nepal|finalist|writer|;Nirajan Rimal|2023|Nepal|finalist|science|;Prakash Badu|2023|Nepal|20under20|child rights|;Prakash Pant|2023|Nepal|20under20|mathematician|;Preeti Pantha|2023|Nepal|finalist|social activist|;Raushan Pandit|2023|Nepal|20under20|technology|;Risham Kumar Sah|2023|Nepal|20under20|environmentalist|;Sadiksha Ghimire|2023|Nepal|20under20|health activist|;Sagar Budha|2023|Nepal|20under20|environmentalist|;Sagar Gupta|2023|Nepal|20under20|AI researcher|;Samip Paudel|2023|Nepal|20under20|technology|;Shubham Upreti|2023|Nepal|20under20|artist|;Sumitra Acharya|2023|Nepal|20under20|activist|;Sushan Shrestha|2023|Nepal|20under20|activist|
# 2024 20under20 (+finalists)
Aadesh Regmi|2024|Nepal|20under20|social activist|;Aashish Panthi|2024|Nepal|finalist|tech education|Kapilvastu;Aayushman Puri|2024|Nepal|20under20|social activist|;Aryan Basnet|2024|Nepal|20under20|IT|;Ashish Banjara|2024|Nepal|20under20|social activist|;Hangsam Nembang|2024|Nepal|20under20|entrepreneur|;Kaushal Niraula|2024|Nepal|20under20|environment|;Kishor Shahi|2024|Nepal|20under20|climate|;Krishtina Khanal|2024|Nepal|20under20|technology|;Nischal Bhattarai|2024|Nepal|finalist|education|Syangja;Prashim Timsina|2024|Nepal|20under20|technopreneur|;Purnima Timsina|2024|Nepal|finalist|child rights|Jhapa;Sajani Sharma|2024|Nepal|finalist|social activism|Kaski;Saurab Banstola|2024|Nepal|20under20|STEM activist|;Shakti K.C.|2024|Nepal|20under20|AI|;Shreejay Subedi|2024|Nepal|finalist|tech|Parsa;Sinshiya K.C.|2024|Nepal|20under20|health activist|;Sugam Parajuli|2024|Nepal|20under20|entrepreneur|;Tushar Shah|2024|Nepal|20under20|innovator|
# 2025 20under20 (+finalists)
Aawish Khanal|2025|Nepal|20under20|entrepreneur|Butwal;Dhurbesh Dhami|2025|Nepal|20under20|campaigner|Bajura;Gokul Shrestha|2025|Nepal|20under20|researcher|Baglung;Manushi Neupane|2025|Nepal|finalist|SRHR advocate|Syangja;Mohammad Aftab Sheikh|2025|Nepal|20under20|entrepreneur|Birgunj;Nischal Dhungana|2025|Nepal|finalist|climate justice|Kapilvastu;Osish Niraula|2025|Nepal|20under20|child rights|Sunsari;Oxford Acharya|2025|Nepal|finalist|policy advocate|Jumla;Phurwa Tsering Gurung|2025|Nepal|20under20|documentary|Dolpo;Pooja Mainali|2025|Nepal|20under20|teen leadership|Jhapa;Rakshit Poudel|2025|Nepal|20under20|STEM educator|Butwal;Renuka Singh|2025|Nepal|finalist|agropreneur|Sarlahi;Ritu Gharti|2025|Nepal|20under20|peer educator|Nawalpur;Ruchi Ojha|2025|Nepal|20under20|menstrual health|Kathmandu;Safal Poudel|2025|Nepal|finalist|AI developer|Rupandehi;Sajan Adhikari|2025|Nepal|20under20|SDG activist|Chitwan;Saksham Ghimire|2025|Nepal|20under20|policy advocate|Rupandehi;Saksham Rupakheti|2025|Nepal|20under20|social entrepreneur|Kapilvastu
"""

def field_tags(f):
    f=f.lower()
    t=set()
    if re.search(r"tech|coder|program|app|develop|it\b|robot|cyber|ai|innovat|engineer|stem|science|space|astronom|math",f): t.add("tech")
    if re.search(r"preneur|entrepreneur|business|startup|intrapreneur",f): t.add("biz")
    if re.search(r"child|rights|activist|social|environment|climate|health|srhr|gender|menstrual|conservation|advocate|sdg|peer|volunteer|campaign|women|education activist|female",f): t.add("social")
    if re.search(r"content|film|document|artist|writer|music|photo|journal|creator|poet",f): t.add("media")
    if re.search(r"educat|mentor|teacher|leadership|olympiad|researcher|research",f): t.add("edu")
    return t

def heuristic(tier,f):
    tags=field_tags(f)
    base = 3.3 if tier=="finalist" else 3.0
    lo   = 2.5 if tier=="finalist" else 2.2
    s={"social_impact":base,"leadership":base,"innovation":lo,"entrepreneurship":lo,
       "recognition":lo+0.5,"glocal_fit":base,"character":base+0.4}
    if "tech" in tags: s["innovation"]+=1.3
    if "biz" in tags:  s["entrepreneurship"]+=1.4; s["leadership"]+=0.3
    if "social" in tags: s["social_impact"]+=1.0; s["leadership"]+=0.3
    if "media" in tags: s["recognition"]+=1.1; s["social_impact"]+=0.3
    if "edu" in tags: s["leadership"]+=0.6; s["social_impact"]+=0.4
    if tags & {"tech","biz"}: s["glocal_fit"]+=0.4
    return {k:round(min(5.0,max(0.0,v)),1) for k,v in s.items()}

# ---- assemble ----
heroes=[]
seen=set()
# winners + hand-scored first
for (name,year),(sc,summ,links) in HAND.items():
    ed = "India" if name=="Rudhvik Dharamkar" else "Sri Lanka" if name=="Risanga Abeygunasekara" else "Bangladesh" if name=="Talha Zubair" else "Nepal"
    award = "Applicant" if year==2026 else "Winner" if (name,year) in WINNER_KEYS else "20under20"
    heroes.append({"name":name,"year":year,"ed":ed,"award":award,"field":"","sum":summ,"s":sc,"est":False,"links":links,"me":year==2026})
    seen.add((name,year))

for line in ROSTER.strip().splitlines():
    line=line.strip()
    if not line or line.startswith("#"): continue
    for rec in line.split(";"):
        parts=[p.strip() for p in rec.split("|")]
        if len(parts)<5: continue
        name,year,ed,tier,field=parts[0],int(parts[1]),parts[2],parts[3],parts[4]
        home=parts[5] if len(parts)>5 else ""
        if (name,year) in seen: continue  # winner/hand overrides win
        seen.add((name,year))
        heroes.append({"name":name,"year":year,"ed":ed,"award":("Finalist" if tier=="finalist" else "20under20"),
                       "field":field,"home":home,"sum":(field.capitalize()+(" · "+home if home else "")),
                       "s":heuristic(tier,field),"est":True,"links":{}})

out={"rubric":RUBRIC,"generated_note":"Winners + Arjan + Nishant Raj Sarraf hand-scored from public records (est=false). All other honorees scored by the documented tier+field heuristic in build.py (est=true) — modeled estimates, not per-person research.","heroes":heroes}
with open(os.path.join(HERE,"data","heroes.json"),"w") as f: json.dump(out,f,indent=1)

# emit corpus.js for the web app (inline, no fetch needed)
with open(os.path.join(HERE,"corpus.js"),"w") as f:
    f.write("window.__CORPUS__="+json.dumps(heroes,separators=(',',':'))+";")

# summary
nw=sum(1 for h in heroes if h["award"]=="Winner")
print(f"heroes: {len(heroes)} | winners: {nw} | finalists: {sum(1 for h in heroes if h['award']=='Finalist')} | 20under20: {sum(1 for h in heroes if h['award']=='20under20')} | hand-scored: {sum(1 for h in heroes if not h['est'])}")
