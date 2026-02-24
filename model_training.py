"""
model_training.py
-----------------
Trains a Logistic Regression model for sentiment classification.
Uses TF-IDF for feature extraction.
Saves the trained model and vectorizer using pickle.

Labels: Positive, Neutral, Negative
"""

import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import preprocess_batch

# ─── Synthetic Training Dataset ─────────────────────────────────────────────

TRAINING_DATA = [
    # ══════════════════════════════════════════════════════
    # POSITIVE — Faculty
    # ══════════════════════════════════════════════════════
    ("The teachers are very knowledgeable and explain concepts clearly", "Positive"),
    ("Faculty is extremely supportive and always available for doubts", "Positive"),
    ("Excellent professors who make learning enjoyable and interactive", "Positive"),
    ("The teaching methodology is innovative and practical", "Positive"),
    ("Faculty guides us well for research and projects", "Positive"),
    ("Teachers are passionate and dedicated to student growth", "Positive"),
    ("Professors are brilliant and inspiring", "Positive"),
    ("The faculty is amazing and very helpful", "Positive"),
    ("Great teachers who really care about students", "Positive"),
    ("Nice faculty overall very supportive", "Positive"),
    ("Good professors explain everything clearly", "Positive"),
    ("Teachers are wonderful and always ready to help", "Positive"),
    ("Lecturers are highly experienced and knowledgeable", "Positive"),
    ("The professors are excellent and very approachable", "Positive"),
    ("Faculty is outstanding best I have seen", "Positive"),

    # POSITIVE — Infrastructure
    ("The campus infrastructure is world-class and very well maintained", "Positive"),
    ("Libraries are well stocked and labs have modern equipment", "Positive"),
    ("The wifi and internet connectivity on campus is excellent", "Positive"),
    ("Hostel facilities are clean and comfortable", "Positive"),
    ("Sports facilities are amazing and well maintained", "Positive"),
    ("Infrastructure supports both academic and extracurricular activities", "Positive"),
    ("Nice infrastructure very clean and modern", "Positive"),
    ("Good infrastructure with well equipped labs", "Positive"),
    ("The campus looks beautiful and is well maintained", "Positive"),
    ("Excellent labs and library facilities", "Positive"),
    ("Infrastructure is great very modern and spacious", "Positive"),
    ("The buildings and classrooms are very nice", "Positive"),
    ("Great campus with amazing sports and gym facilities", "Positive"),
    ("Infrastructure is top class everything is well maintained", "Positive"),
    ("The college has wonderful infrastructure and facilities", "Positive"),

    # POSITIVE — Curriculum
    ("The curriculum is industry-relevant and up to date", "Positive"),
    ("The syllabus covers all essential topics for our career", "Positive"),
    ("The assignment and project work really builds skills", "Positive"),
    ("Good curriculum that prepares us for industry", "Positive"),
    ("Excellent syllabus very well structured and modern", "Positive"),
    ("The course content is very useful and practical", "Positive"),
    ("Curriculum is great covers everything needed", "Positive"),
    ("The academic program is excellent and well designed", "Positive"),
    ("Good course structure with practical and theory balance", "Positive"),
    ("The syllabus is updated regularly which is very good", "Positive"),

    # POSITIVE — Placements
    ("Placements are outstanding, top companies visit every year", "Positive"),
    ("Great job opportunities offered through campus recruitment drives", "Positive"),
    ("Highest salary packages offered during placements", "Positive"),
    ("The placement cell works tirelessly for students", "Positive"),
    ("Career guidance and internship support is excellent", "Positive"),
    ("Excellent placements with very high salary packages", "Positive"),
    ("Good placement record many students got jobs", "Positive"),
    ("The college has great placement opportunities", "Positive"),
    ("Placements are amazing top MNCs visit campus", "Positive"),
    ("Nice placement support from the college", "Positive"),

    # POSITIVE — Management
    ("Management is very responsive and student-friendly policies", "Positive"),
    ("College administration handles grievances efficiently", "Positive"),
    ("Management organizes excellent events and fests", "Positive"),
    ("Management listens to student feedback and acts on it", "Positive"),
    ("The administration is very supportive and helpful", "Positive"),
    ("Good management that cares about student welfare", "Positive"),
    ("Excellent management and very well organized college", "Positive"),
    ("The management team is proactive and student friendly", "Positive"),

    # POSITIVE — General
    ("Overall a very positive and enriching learning environment", "Positive"),
    ("Very good college overall, highly recommend it", "Positive"),
    ("Canteen food is tasty and hygienic", "Positive"),
    ("College fees are reasonable for the quality provided", "Positive"),
    ("The academic schedule is well planned and organized", "Positive"),
    ("Excellent college experience overall very happy", "Positive"),
    ("This is a great college I love studying here", "Positive"),
    ("The college is wonderful in every aspect", "Positive"),
    ("Very happy with the college experience overall", "Positive"),
    ("Brilliant college highly recommend to everyone", "Positive"),
    ("The experience here has been fantastic and rewarding", "Positive"),
    ("Superb college with great environment for learning", "Positive"),
    ("Amazing college best decision of my life", "Positive"),
    ("Really good college with excellent facilities", "Positive"),
    ("Wonderful learning experience at this institution", "Positive"),

    # ══════════════════════════════════════════════════════
    # NEGATIVE — Faculty
    # ══════════════════════════════════════════════════════
    ("The teachers are not well prepared and often skip topics", "Negative"),
    ("Faculty is rarely available and does not help with doubts", "Negative"),
    ("Poor teaching quality, professors just read from slides", "Negative"),
    ("Faculty shows bias and does not treat all students equally", "Negative"),
    ("Teachers are arrogant and unapproachable", "Negative"),
    ("The professors are terrible and do not know their subjects", "Negative"),
    ("Bad faculty who never explains anything properly", "Negative"),
    ("Teachers are very rude and unhelpful", "Negative"),
    ("Faculty is useless they just waste our time", "Negative"),
    ("The lecturers are boring and do not engage students", "Negative"),
    ("Professors are not qualified and teach incorrectly", "Negative"),
    ("Teachers always come late and leave early very irresponsible", "Negative"),
    ("Faculty does not care about students at all", "Negative"),
    ("The teaching is very poor and hard to understand", "Negative"),
    ("Bad teachers who are never available for help", "Negative"),

    # NEGATIVE — Infrastructure
    ("Infrastructure is terrible, classrooms are dirty and broken", "Negative"),
    ("The lab equipment is outdated and often non-functional", "Negative"),
    ("Library has very few books and resources are inadequate", "Negative"),
    ("Wifi doesn't work and internet connectivity is pathetic", "Negative"),
    ("Hostel facilities are unhygienic and poorly maintained", "Negative"),
    ("Sports facilities are non-existent and neglected", "Negative"),
    ("Infrastructure is crumbling and needs immediate attention", "Negative"),
    ("The campus is very dirty and poorly maintained", "Negative"),
    ("Bad infrastructure classrooms have broken furniture", "Negative"),
    ("The labs are useless with outdated computers and equipment", "Negative"),
    ("Poor infrastructure very old and dilapidated buildings", "Negative"),
    ("The library is very small and lacks proper resources", "Negative"),
    ("Infrastructure is awful nothing works properly", "Negative"),
    ("Terrible facilities the campus is a mess", "Negative"),
    ("The hostel and canteen are absolutely disgusting", "Negative"),

    # NEGATIVE — Curriculum
    ("Curriculum is outdated and not aligned with industry needs", "Negative"),
    ("No practical sessions, only theoretical teaching", "Negative"),
    ("The syllabus is old and irrelevant for modern careers", "Negative"),
    ("Assignments are repetitive and do not build real skills", "Negative"),
    ("The course content is very poor and outdated", "Negative"),
    ("Bad curriculum that does not prepare us for jobs", "Negative"),
    ("The syllabus is useless and not updated for years", "Negative"),
    ("Poor course structure with no practical exposure at all", "Negative"),
    ("The academic program is badly designed and confusing", "Negative"),
    ("Curriculum is terrible and a waste of time", "Negative"),

    # NEGATIVE — Placements
    ("Placements are very poor, only low-paying companies come", "Negative"),
    ("Very few students get jobs through campus recruitment", "Negative"),
    ("Placement packages are very low compared to other colleges", "Negative"),
    ("The placement cell makes false promises to students", "Negative"),
    ("No career guidance or internship support provided", "Negative"),
    ("Terrible placements very few companies visit campus", "Negative"),
    ("Bad placement record most students are unemployed", "Negative"),
    ("The placement support is useless and unhelpful", "Negative"),
    ("Placements are a joke no good companies ever come", "Negative"),
    ("Poor placement opportunities very disappointed", "Negative"),

    # NEGATIVE — Management
    ("Management is rigid, unresponsive and does not care about students", "Negative"),
    ("College administration is corrupt and mismanages funds", "Negative"),
    ("Management cancels events without prior notice", "Negative"),
    ("Management ignores student complaints and feedback", "Negative"),
    ("The administration is very strict and student unfriendly", "Negative"),
    ("Bad management that only cares about fees and money", "Negative"),
    ("Terrible administration with no transparency", "Negative"),
    ("The management is horrible and does not listen to students", "Negative"),
    ("College fees are exorbitant for the poor quality provided", "Negative"),
    ("Management is corrupt and very poorly organized", "Negative"),

    # NEGATIVE — General
    ("Very poor college experience, would not recommend", "Negative"),
    ("Academic schedule changes frequently causing confusion", "Negative"),
    ("Overall a very disappointing and frustrating experience", "Negative"),
    ("This college is terrible do not join here", "Negative"),
    ("Worst college experience ever very unhappy", "Negative"),
    ("The college is a waste of time and money", "Negative"),
    ("Very bad experience here would not recommend to anyone", "Negative"),
    ("Pathetic college with no redeeming qualities", "Negative"),
    ("The experience here was horrible and demotivating", "Negative"),
    ("Terrible college avoid it at all costs", "Negative"),
    ("Canteen food is unhealthy and overpriced", "Negative"),
    ("The college is poorly managed and very disappointing", "Negative"),
    ("Nothing works here it is a complete disaster", "Negative"),
    ("Very frustrating experience at this institution", "Negative"),
    ("The worst decision was joining this college", "Negative"),

    # ══════════════════════════════════════════════════════
    # NEUTRAL
    # ══════════════════════════════════════════════════════
    ("The college is average, neither too good nor too bad", "Neutral"),
    ("Teachers are okay, some are good and some need improvement", "Neutral"),
    ("Infrastructure is decent but could use some upgrades", "Neutral"),
    ("Curriculum is standard, nothing exceptional", "Neutral"),
    ("Placements are average compared to other colleges in the area", "Neutral"),
    ("Management is neither very good nor very bad", "Neutral"),
    ("Faculty quality varies from department to department", "Neutral"),
    ("Lab facilities are acceptable for basic practical work", "Neutral"),
    ("The syllabus covers basics but lacks advanced topics", "Neutral"),
    ("Some companies visit for placements but packages are moderate", "Neutral"),
    ("Administration handles most issues but is sometimes slow", "Neutral"),
    ("College is okay for the fees being charged", "Neutral"),
    ("Internet works sometimes but is inconsistent", "Neutral"),
    ("Hostel is livable but basic amenities could be better", "Neutral"),
    ("Assignments are fine, not too difficult or too easy", "Neutral"),
    ("Faculty explains concepts adequately for exam purposes", "Neutral"),
    ("Campus is clean enough for day to day activities", "Neutral"),
    ("The placement statistics are in line with industry averages", "Neutral"),
    ("Management responds to major complaints but ignores minor ones", "Neutral"),
    ("Overall an average experience, met basic expectations", "Neutral"),
    ("Some lectures are engaging while others are boring", "Neutral"),
    ("Library has enough resources for basic study needs", "Neutral"),
    ("The academic program is standard for this type of college", "Neutral"),
    ("Canteen food is acceptable but not great", "Neutral"),
    ("The college has potential but needs improvement in several areas", "Neutral"),
    ("Teaching is satisfactory, though not inspiring", "Neutral"),
    ("Infrastructure serves its purpose but lacks modern amenities", "Neutral"),
    ("Placements happen but students have to put in extra effort", "Neutral"),
    ("Management policies are reasonable but somewhat inflexible", "Neutral"),
    ("The college experience has been a mixed bag overall", "Neutral"),
    ("Faculty is okay not great but not bad either", "Neutral"),
    ("The campus is alright nothing special about it", "Neutral"),
    ("College is decent for the location and fees", "Neutral"),
    ("The curriculum is passable but could be more industry focused", "Neutral"),
    ("Placements are so so neither good nor bad", "Neutral"),
    ("The management is average could do better", "Neutral"),
    ("Infrastructure is moderate suits basic needs", "Neutral"),
    ("Some teachers are good some are not so good", "Neutral"),
    ("The college is fine overall nothing to complain much about", "Neutral"),
    ("Average experience with some good and some bad aspects", "Neutral"),

    # ══════════════════════════════════════════════════════
    # LIGHT HUMOUR — sounds like a joke but means POSITIVE
    # The student is happy, expressing it humorously
    # ══════════════════════════════════════════════════════
    ("The teacher explains so well even my brain attends class", "Positive"),
    ("Faculty is so good I actually want to come to college", "Positive"),
    ("Professor makes boring topics interesting which is basically magic", "Positive"),
    ("The campus is so beautiful I forget I am here to study", "Positive"),
    ("Infrastructure is so good I sometimes forget to go home", "Positive"),
    ("The library is so peaceful I could live here honestly", "Positive"),
    ("Placement cell actually replies which is shocking in a good way", "Positive"),
    ("Canteen food is so good I eat more than I study", "Positive"),
    ("The wifi is so fast I thought I was dreaming", "Positive"),
    ("College is so well managed I keep checking if this is real", "Positive"),
    ("Teachers are so helpful I feel guilty not studying", "Positive"),
    ("The labs are so modern even the equipment looks surprised", "Positive"),
    ("Placements are so good my parents finally stopped worrying", "Positive"),
    ("Management actually listens to students which honestly shocked me", "Positive"),
    ("The hostel food improved so much I stopped ordering outside", "Positive"),

    # ══════════════════════════════════════════════════════
    # SARCASM — sounds positive but means NEGATIVE
    # Key signal: praise followed by contradiction or absurd qualifier
    # ══════════════════════════════════════════════════════
    ("Great wifi works perfectly when the class ends", "Negative"),
    ("Love the ac it teaches survival skills in summer", "Negative"),
    ("Placement training is amazing if your goal is patience", "Negative"),
    ("Teachers are so helpful they give assignments on holidays too", "Negative"),
    ("The library is fantastic if you enjoy staring at empty shelves", "Negative"),
    ("Management is super responsive they replied after only three months", "Negative"),
    ("Amazing infrastructure if you enjoy broken chairs and leaking roofs", "Negative"),
    ("Great placements only three students got placed out of hundred", "Negative"),
    ("Wonderful curriculum prepares you for jobs that stopped existing", "Negative"),
    ("Excellent canteen food if you enjoy food poisoning occasionally", "Negative"),
    ("Brilliant professors who read directly from ten year old slides", "Negative"),
    ("Fantastic college experience if you enjoy wasting time and money", "Negative"),
    ("The wifi is amazing it connects for exactly five minutes a day", "Negative"),
    ("Love how the lab computers boot up in only thirty minutes", "Negative"),
    ("The management is so efficient it took a year to fix one light", "Negative"),
    ("What a great college the ceiling fan works only during winter", "Negative"),
    ("Placement cell is very active at making excuses for poor results", "Negative"),
    ("Curriculum is very modern we study technologies from twenty years ago", "Negative"),

    # ══════════════════════════════════════════════════════
    # DARK HUMOUR — no obvious negative words but NEGATIVE
    # Uses exaggeration, absurdity, or existential dread
    # ══════════════════════════════════════════════════════
    ("Assignments make me question my life choices every single day", "Negative"),
    ("Labs are so outdated they probably belong in a museum", "Negative"),
    ("Attendance policy is stricter than my parents ever were", "Negative"),
    ("Studying here made me rediscover the joy of doing nothing", "Negative"),
    ("The timetable changes so often I stopped keeping track", "Negative"),
    ("After four years here I am an expert at pretending to learn", "Negative"),
    ("The exams test how well you memorize things you will never use", "Negative"),
    ("I learned more about patience here than any actual subject", "Negative"),
    ("College has taught me that suffering builds character apparently", "Negative"),
    ("Every semester I discover a new way this place disappoints me", "Negative"),
    ("The rules here make me nostalgic for freedom", "Negative"),
    ("Fees go up every year quality stays exactly where it was in 2005", "Negative"),
    ("I have grown a lot here mainly in frustration and confusion", "Negative"),
    ("The college prepares you for the real world by already feeling hopeless", "Negative"),
    ("Nothing here works but at least the experience is consistently bad", "Negative"),

    # ══════════════════════════════════════════════════════
    # DARK SARCASM — most deceptive, hardest to detect
    # Sounds positive on surface, deeply negative underneath
    # ══════════════════════════════════════════════════════
    ("Faculty is supportive emotionally after destroying us academically", "Negative"),
    ("Placement cell prepares you very well for unemployment", "Negative"),
    ("Curriculum prepares you for jobs that do not exist yet or ever", "Negative"),
    ("The college builds your character by systematically breaking your spirit", "Negative"),
    ("Management is very invested in collecting fees less so in education", "Negative"),
    ("Teachers inspire you to study on your own because they certainly won't help", "Negative"),
    ("The college gives you four years to figure out it was a mistake", "Negative"),
    ("Professors are very consistent consistently unavailable and unhelpful", "Negative"),
    ("Infrastructure is historical in the sense that nothing has changed in decades", "Negative"),
    ("The placement statistics are impressive if you count rejections", "Negative"),
    ("Faculty is dedicated to finishing syllabus not to teaching it", "Negative"),
    ("Management cares deeply about discipline much less about student welfare", "Negative"),
    ("The college has a rich tradition of overpromising and underdelivering", "Negative"),
    ("Labs are well equipped for the 1990s admittedly", "Negative"),
    ("The fees teach you the valuable lesson that money cannot buy quality", "Negative"),

    # ══════════════════════════════════════════════════════
    # MIXED SENTIMENT examples
    # These help the model understand contrast words
    # ══════════════════════════════════════════════════════
    ("Faculty is amazing but infrastructure is terrible and needs fixing", "Negative"),
    ("Curriculum is good but placements are very disappointing", "Negative"),
    ("Campus is beautiful but wifi is absolutely tragic", "Negative"),
    ("Teachers are excellent however the labs are outdated and useless", "Negative"),
    ("Good faculty but management is completely unresponsive", "Negative"),
    ("Infrastructure is great but curriculum is outdated and irrelevant", "Negative"),
    ("Placements improved but faculty quality has dropped significantly", "Negative"),
    ("The college looks good from outside but inside it is a mess", "Negative"),
    ("Some good teachers but overall the experience has been disappointing", "Negative"),
    ("Nice campus but everything else about this college is below average", "Negative"),

    # ══════════════════════════════════════════════════════
    # FACULTY — Short clean examples
    # ══════════════════════════════════════════════════════
    # Positive
    ("The faculty explains concepts very clearly", "Positive"),
    ("Teachers are supportive and approachable", "Positive"),
    ("Faculty makes difficult topics easy to understand", "Positive"),
    ("Professors genuinely care about students", "Positive"),
    ("Teaching quality is excellent", "Positive"),
    ("The faculty is very dedicated and hardworking", "Positive"),
    ("Teachers always encourage students to ask questions", "Positive"),
    ("Faculty provides very clear and structured explanations", "Positive"),
    ("Professors are experienced and teach very effectively", "Positive"),
    ("The teaching staff is highly skilled and helpful", "Positive"),
    # Neutral
    ("Faculty follows the syllabus as given", "Neutral"),
    ("Teachers complete lectures on time", "Neutral"),
    ("Classes happen regularly without major issues", "Neutral"),
    ("Teaching is standard nothing special", "Neutral"),
    ("Faculty performance is average overall", "Neutral"),
    ("Professors cover required topics adequately", "Neutral"),
    ("Teachers are neither too good nor too bad", "Neutral"),
    ("Faculty is okay does the basic job", "Neutral"),
    ("Teaching style is conventional and acceptable", "Neutral"),
    ("Faculty is average compared to other colleges", "Neutral"),
    # Negative
    ("Faculty reads slides without explaining anything", "Negative"),
    ("Teachers are not available after class for help", "Negative"),
    ("Lectures are boring and hard to sit through", "Negative"),
    ("Doubt solving is very poor and unhelpful", "Negative"),
    ("Faculty is inconsistent in teaching and attendance", "Negative"),
    ("Professors do not prepare for lectures at all", "Negative"),
    ("Teachers ignore student questions during class", "Negative"),
    ("Faculty shows favouritism and is very unprofessional", "Negative"),
    ("Teaching is rushed and topics are never explained properly", "Negative"),
    ("The faculty lacks subject knowledge and expertise", "Negative"),

    # ══════════════════════════════════════════════════════
    # INFRASTRUCTURE — Short clean examples
    # ══════════════════════════════════════════════════════
    # Positive
    ("Campus infrastructure is very modern and updated", "Positive"),
    ("Labs are well equipped with latest technology", "Positive"),
    ("Classrooms are comfortable and well ventilated", "Positive"),
    ("Library has very good resources and books", "Positive"),
    ("Facilities are well maintained and clean", "Positive"),
    ("The campus is spacious and beautifully designed", "Positive"),
    ("Sports facilities and gym are excellent", "Positive"),
    ("Hostel rooms are clean and comfortable", "Positive"),
    ("Campus wifi is fast and reliable", "Positive"),
    ("The canteen is hygienic and serves good food", "Positive"),
    # Neutral
    ("Infrastructure is average nothing exceptional", "Neutral"),
    ("Labs are functional for basic needs", "Neutral"),
    ("Facilities are usable but could be better", "Neutral"),
    ("Campus is decent for the size of the college", "Neutral"),
    ("Infrastructure is manageable for day to day use", "Neutral"),
    ("The buildings are okay but need some renovation", "Neutral"),
    ("Library is adequate for basic study requirements", "Neutral"),
    ("Facilities are passable but not impressive", "Neutral"),
    ("Campus is clean enough but lacks modern amenities", "Neutral"),
    ("Infrastructure is standard for this type of institution", "Neutral"),
    # Negative
    ("Labs have outdated computers that barely work", "Negative"),
    ("WiFi barely works and keeps disconnecting", "Negative"),
    ("Classrooms are overcrowded and very uncomfortable", "Negative"),
    ("Equipment is broken and never gets repaired", "Negative"),
    ("Library lacks books and updated study materials", "Negative"),
    ("The campus is dirty and poorly maintained", "Negative"),
    ("Hostel facilities are unhygienic and run down", "Negative"),
    ("No proper sports facilities available at all", "Negative"),
    ("Canteen food is very poor quality and unhealthy", "Negative"),
    ("The college buildings are very old and crumbling", "Negative"),

    # ══════════════════════════════════════════════════════
    # CURRICULUM — Short clean examples
    # ══════════════════════════════════════════════════════
    # Positive
    ("Curriculum is very industry relevant and modern", "Positive"),
    ("Subjects are interesting and practically useful", "Positive"),
    ("Course structure is very well balanced", "Positive"),
    ("Practical exposure given is very good", "Positive"),
    ("Syllabus is well designed and comprehensive", "Positive"),
    ("The course content keeps us industry ready", "Positive"),
    ("Subjects taught are directly applicable to real jobs", "Positive"),
    ("Curriculum includes both theory and hands on projects", "Positive"),
    ("Course is updated regularly to match industry trends", "Positive"),
    ("The syllabus prepares us very well for placements", "Positive"),
    # Neutral
    ("Curriculum is manageable and not too difficult", "Neutral"),
    ("Subjects are standard as expected for this level", "Neutral"),
    ("Syllabus is typical for this type of program", "Neutral"),
    ("Course difficulty is average and fair", "Neutral"),
    ("Curriculum is moderate suits basic learning needs", "Neutral"),
    ("The course covers required topics nothing more", "Neutral"),
    ("Syllabus is okay not too heavy or too light", "Neutral"),
    ("Course structure follows a standard pattern", "Neutral"),
    ("Subjects are passable though not very exciting", "Neutral"),
    ("The curriculum meets minimum academic requirements", "Neutral"),
    # Negative
    ("Curriculum is very outdated and not relevant", "Negative"),
    ("Too much theory and no practical application at all", "Negative"),
    ("No real world skills are taught in this course", "Negative"),
    ("Subjects feel completely irrelevant to industry", "Negative"),
    ("Course is repetitive and covers same content each year", "Negative"),
    ("Syllabus has not been updated in many years", "Negative"),
    ("The course does not prepare students for actual jobs", "Negative"),
    ("Too many unnecessary subjects waste our time", "Negative"),
    ("Practical sessions are almost non existent in this program", "Negative"),
    ("The curriculum is poorly designed and hard to follow", "Negative"),

    # ══════════════════════════════════════════════════════
    # PLACEMENTS — Short clean examples
    # ══════════════════════════════════════════════════════
    # Positive
    ("Placement support is very strong and active", "Positive"),
    ("Many top companies visit campus for recruitment", "Positive"),
    ("Good internship opportunities are provided regularly", "Positive"),
    ("Training sessions before placements are very helpful", "Positive"),
    ("Placement guidance and support is very useful", "Positive"),
    ("The placement cell is dedicated and works hard", "Positive"),
    ("High salary packages are offered through campus placements", "Positive"),
    ("Most students get placed through campus drives", "Positive"),
    ("Career counselling and placement prep is excellent", "Positive"),
    ("The college has strong industry connections for placements", "Positive"),
    # Neutral
    ("Placement cell shares updates and job listings", "Neutral"),
    ("Some companies visit campus for hiring", "Neutral"),
    ("Placement record is average compared to peers", "Neutral"),
    ("Opportunities are moderate neither great nor bad", "Neutral"),
    ("Placement process is standard and organised", "Neutral"),
    ("A few students manage to get placed each year", "Neutral"),
    ("Placement support exists but could be more proactive", "Neutral"),
    ("The placement statistics are in line with industry average", "Neutral"),
    ("Internship support is available but limited", "Neutral"),
    ("Campus recruitment happens but mostly mid tier companies", "Neutral"),
    # Negative
    ("Very few companies visit campus for placement", "Negative"),
    ("Placement training is very weak and insufficient", "Negative"),
    ("Salary offers through campus are very low", "Negative"),
    ("No core companies visit for campus recruitment", "Negative"),
    ("Placement support is very poor and unhelpful", "Negative"),
    ("Most students have to find jobs on their own", "Negative"),
    ("The placement cell is inactive and disorganised", "Negative"),
    ("No proper interview preparation is provided", "Negative"),
    ("Placement opportunities are almost zero for many branches", "Negative"),
    ("The college placement record is very disappointing", "Negative"),

    # ══════════════════════════════════════════════════════
    # MANAGEMENT — Short clean examples
    # ══════════════════════════════════════════════════════
    # Positive
    ("Administration is very helpful and cooperative", "Positive"),
    ("Management responds to student issues very quickly", "Positive"),
    ("Administrative processes are smooth and efficient", "Positive"),
    ("Support staff is cooperative and friendly", "Positive"),
    ("Communication from management is very clear", "Positive"),
    ("The management team genuinely cares about students", "Positive"),
    ("College admin handles complaints and requests well", "Positive"),
    ("Management organises events and activities regularly", "Positive"),
    ("Policies are fair and student friendly", "Positive"),
    ("The administration is transparent and well organised", "Positive"),
    # Neutral
    ("Management is okay handles basic tasks", "Neutral"),
    ("Administration works normally without major issues", "Neutral"),
    ("Rules and regulations are standard and expected", "Neutral"),
    ("Administrative processes are average in speed", "Neutral"),
    ("Support from management is moderate and acceptable", "Neutral"),
    ("College admin does its job without going above and beyond", "Neutral"),
    ("Management is neither very strict nor very lenient", "Neutral"),
    ("The administration handles most things adequately", "Neutral"),
    ("Communication from admin is occasional and basic", "Neutral"),
    ("Management policies are standard for this type of college", "Neutral"),
    # Negative
    ("Administration is very slow in resolving issues", "Negative"),
    ("Communication from management is very poor", "Negative"),
    ("Issues take months to get resolved by admin", "Negative"),
    ("Rules change very frequently causing confusion", "Negative"),
    ("Support from management is completely lacking", "Negative"),
    ("The administration is unresponsive and bureaucratic", "Negative"),
    ("Management ignores student complaints entirely", "Negative"),
    ("College admin is corrupt and mismanages resources", "Negative"),
    ("Policies are unfair and heavily favour management", "Negative"),
    ("The administration creates more problems than it solves", "Negative"),

    # ══════════════════════════════════════════════════════
    # SLANG / INFORMAL / SHORT WORDS — Gen Z and casual feedback
    # These teach the model modern informal language patterns
    # ══════════════════════════════════════════════════════

    # ── Single word positive slang ──
    ("lit", "Positive"),
    ("banger", "Positive"),
    ("fire", "Positive"),
    ("solid", "Positive"),
    ("dope", "Positive"),
    ("clutch", "Positive"),
    ("impressive", "Positive"),
    ("clean", "Positive"),
    ("smooth", "Positive"),
    ("nice", "Positive"),
    ("decent", "Positive"),

    # ── Short phrase positive slang ──
    ("top notch", "Positive"),
    ("on point", "Positive"),
    ("worth it", "Positive"),
    ("great stuff", "Positive"),
    ("loved it", "Positive"),
    ("works well", "Positive"),
    ("super helpful", "Positive"),
    ("amazing experience", "Positive"),
    ("really good", "Positive"),
    ("pretty great", "Positive"),
    ("absolutely loved it", "Positive"),
    ("highly recommend", "Positive"),
    ("exceeded expectations", "Positive"),
    ("so good", "Positive"),
    ("nailed it", "Positive"),
    ("killed it", "Positive"),
    ("straight up excellent", "Positive"),
    ("no complaints", "Positive"),
    ("actually impressive", "Positive"),
    ("surprisingly good", "Positive"),

    # ── Slang in college context — positive ──
    ("faculty is lit", "Positive"),
    ("classes are fire", "Positive"),
    ("professor is a banger", "Positive"),
    ("campus is dope", "Positive"),
    ("placement was clutch", "Positive"),
    ("labs are on point", "Positive"),
    ("curriculum is solid", "Positive"),
    ("management is top notch", "Positive"),
    ("infrastructure is clean", "Positive"),
    ("teaching is smooth", "Positive"),
    ("college is worth it", "Positive"),
    ("faculty is super helpful", "Positive"),
    ("placements are fire", "Positive"),
    ("this college is a banger", "Positive"),
    ("campus vibes are great", "Positive"),
    ("everything is on point here", "Positive"),
    ("really decent college overall", "Positive"),
    ("honestly pretty impressive", "Positive"),

    # ── Single word neutral slang ──
    ("okay", "Neutral"),
    ("fine", "Neutral"),
    ("average", "Neutral"),
    ("manageable", "Neutral"),
    ("fair", "Neutral"),
    ("acceptable", "Neutral"),
    ("normal", "Neutral"),
    ("standard", "Neutral"),
    ("typical", "Neutral"),
    ("works", "Neutral"),
    ("alright", "Neutral"),
    ("meh", "Neutral"),
    ("mid", "Neutral"),

    # ── Short phrase neutral slang ──
    ("not bad", "Neutral"),
    ("not great", "Neutral"),
    ("could be better", "Neutral"),
    ("decent enough", "Neutral"),
    ("nothing special", "Neutral"),
    ("does the job", "Neutral"),
    ("gets the work done", "Neutral"),
    ("so so", "Neutral"),
    ("kind of okay", "Neutral"),
    ("pretty average", "Neutral"),
    ("not bad not great", "Neutral"),
    ("somewhere in the middle", "Neutral"),
    ("neither good nor bad", "Neutral"),
    ("okay i guess", "Neutral"),
    ("it is what it is", "Neutral"),
    ("nothing to write home about", "Neutral"),
    ("meets expectations barely", "Neutral"),
    ("exists i suppose", "Neutral"),

    # ── Neutral in college context ──
    ("faculty is okay", "Neutral"),
    ("college is fine i guess", "Neutral"),
    ("curriculum is not bad", "Neutral"),
    ("infrastructure is decent enough", "Neutral"),
    ("placements are average", "Neutral"),
    ("management is manageable", "Neutral"),
    ("college is pretty mid honestly", "Neutral"),
    ("teaching is fair nothing special", "Neutral"),
    ("labs are acceptable for basic work", "Neutral"),
    ("overall it is just okay", "Neutral"),

    # ── Single word negative slang ──
    ("trash", "Negative"),
    ("weak", "Negative"),
    ("useless", "Negative"),
    ("terrible", "Negative"),
    ("awful", "Negative"),
    ("disappointing", "Negative"),
    ("outdated", "Negative"),
    ("frustrating", "Negative"),
    ("slow", "Negative"),
    ("buggy", "Negative"),
    ("messy", "Negative"),
    ("poor", "Negative"),
    ("waste", "Negative"),
    ("painful", "Negative"),
    ("exhausting", "Negative"),
    ("stressful", "Negative"),
    ("draining", "Negative"),
    ("confusing", "Negative"),
    ("chaotic", "Negative"),
    ("pointless", "Negative"),
    ("unbearable", "Negative"),
    ("ridiculous", "Negative"),
    ("nightmare", "Negative"),

    # ── Short phrase negative slang ──
    ("sucks", "Negative"),
    ("not good", "Negative"),
    ("very bad", "Negative"),
    ("needs improvement", "Negative"),
    ("worst experience", "Negative"),
    ("doesn't help", "Negative"),
    ("no support", "Negative"),
    ("not useful", "Negative"),
    ("not helpful", "Negative"),
    ("not clear", "Negative"),
    ("tests patience", "Negative"),
    ("waste of time", "Negative"),
    ("makes no sense", "Negative"),
    ("outdated stuff", "Negative"),
    ("nothing works", "Negative"),
    ("absolutely terrible", "Negative"),
    ("complete waste", "Negative"),
    ("total disaster", "Negative"),
    ("genuinely awful", "Negative"),
    ("deeply disappointing", "Negative"),
    ("beyond frustrating", "Negative"),
    ("straight up trash", "Negative"),
    ("absolutely pointless", "Negative"),
    ("so stressful", "Negative"),
    ("so draining", "Negative"),
    ("zero value", "Negative"),
    ("pure chaos", "Negative"),

    # ── Negative slang in college context ──
    ("faculty is trash", "Negative"),
    ("teaching is useless", "Negative"),
    ("college is a nightmare", "Negative"),
    ("infrastructure is outdated stuff", "Negative"),
    ("management is chaotic", "Negative"),
    ("placements are a complete waste", "Negative"),
    ("curriculum is pointless", "Negative"),
    ("labs are buggy and messy", "Negative"),
    ("wifi is terrible and slow", "Negative"),
    ("exams are so draining and stressful", "Negative"),
    ("faculty is so frustrating", "Negative"),
    ("classes are absolutely exhausting", "Negative"),
    ("management is pure chaos", "Negative"),
    ("the whole system is confusing", "Negative"),
    ("this college is a total disaster", "Negative"),
    ("placements are trash honestly", "Negative"),
    ("everything here is so outdated", "Negative"),
    ("administration is beyond frustrating", "Negative"),
    ("nothing works here it is a nightmare", "Negative"),
    ("fees are ridiculous for this quality", "Negative"),
]


def prepare_dataset(data: list) -> tuple:
    """
    Extract texts and labels from the training dataset.

    Returns:
        tuple: (list of texts, list of labels)
    """
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    return texts, labels


def train_model(save_path: str = "model") -> dict:
    """
    Train TF-IDF + Logistic Regression pipeline and save to disk.

    Parameters:
        save_path (str): Directory path to save model artifacts.

    Returns:
        dict: Training metrics including accuracy and classification report.
    """
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Prepare raw dataset
    texts, labels = prepare_dataset(TRAINING_DATA)

    # Step 2: Preprocess texts
    print("Preprocessing training texts...")
    processed_texts = preprocess_batch(texts)

    # Step 3: Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Step 4: TF-IDF Vectorization
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 3),      # Unigrams + bigrams + trigrams
        min_df=1,
        sublinear_tf=True,       # Apply log normalization
        analyzer='word',
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only real words
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 5: Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=2000,
        C=5.0,                   # Higher C = less regularization = better fit
        class_weight='balanced',
        random_state=42,
        solver='lbfgs',
        multi_class='multinomial'
    )
    model.fit(X_train_tfidf, y_train)

    # Step 6: Evaluate on test set
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n✅ Model trained successfully!")
    print(f"   Test Accuracy: {accuracy:.2%}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Step 7: Save model and vectorizer
    model_file = os.path.join(save_path, "sentiment_model.pkl")
    vectorizer_file = os.path.join(save_path, "tfidf_vectorizer.pkl")

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\n💾 Model saved to: {model_file}")
    print(f"💾 Vectorizer saved to: {vectorizer_file}")

    return {
        "accuracy": accuracy,
        "report": report,
        "model_path": model_file,
        "vectorizer_path": vectorizer_file,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }


def load_model(model_path: str = "model/sentiment_model.pkl",
               vectorizer_path: str = "model/tfidf_vectorizer.pkl") -> tuple:
    """
    Load a pre-trained model and vectorizer from disk.

    Returns:
        tuple: (model, vectorizer)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


if __name__ == "__main__":
    # Run training when script is executed directly
    metrics = train_model()
    print(f"\nFinal Accuracy: {metrics['accuracy']:.2%}")
