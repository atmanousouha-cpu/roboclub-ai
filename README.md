# RoboClub AI – An AI coach for middle-school robotics clubs

Final project for the Building AI course

## Summary

RoboClub AI is a learning assistant for middle-school robotics clubs. It suggests a robotics project adapted to each team's level and interests, helps students debug their robots, and lets them train small AI models (image and sound recognition) that their robots can use.

## Background

I teach technology in a public middle school (collège) in Morocco and I am a master's student in Educational Technology Engineering and Innovation (Faculty of Educational Sciences, Mohammed V University, Rabat). Robotics clubs motivate students a lot, but running them is hard:

* **One teacher, many teams**: a single teacher has to support 20–30 students working on different robots at the same time.
* **Mixed levels**: some students have never programmed, others already know Scratch or Arduino. One project for everybody is too hard for some and boring for others.
* **Students get stuck**: a robot that doesn't move is often caused by a small wiring or code error, and students give up before the teacher can help.
* **AI is in the curriculum, but abstract**: students hear about AI but rarely build anything with it themselves.
* **Limited resources**: few kits, unstable internet, and documentation mostly in English.

My motivation is to make robotics clubs more inclusive and to let students learn AI by *doing*, not only by listening. This matches the national goals of digital transformation in education and develops 21st-century skills (problem solving, collaboration, creativity).

## How is it used?

The solution is used during weekly club sessions (1–2 hours), in the technology lab, on the teacher's computer or a few shared tablets.

**1. Getting started – project recommendation**
Each team fills in a short questionnaire (age, programming experience, interests like sports, environment, music, available kit). RoboClub AI suggests 3 projects of the right difficulty, e.g. *line-following robot*, *automatic plant-watering system*, *robot that sorts recyclable waste by color*.

**2. Building – "help me" assistant**
When the robot doesn't work, students describe the problem in simple words (Arabic, French or Darija), e.g. "the motor turns only in one direction". The assistant asks guiding questions and gives hints step by step, instead of giving the full solution.

**3. Adding AI to the robot – train your own model**
Students collect their own data (photos of colored objects, hand gestures, voice commands) with a webcam, train a small classifier, and send the result to their Arduino/micro:bit robot. Example: a robot that stops when it "sees" a red card.

**4. Following progress – teacher dashboard**
The teacher sees which teams are stuck, which skills each student practiced, and gets suggestions for the next session.

**Users:**

* Students aged 12–15, beginners in programming
* Technology teachers who lead the club
* Optionally parents and school management (end-of-year exhibition)

Needs to take into account: simple interface, multilingual (Arabic / French), works offline, low-cost hardware.

Example: recommending a project with the **nearest neighbor method** (Chapter 3 of the course). Each team and each project is described by the same features, and we recommend the projects closest to the team's profile:

```python
import numpy as np

# features: [programming level 0-3, mechanics level 0-3, interest in environment 0-1, interest in games 0-1, needs camera 0-1]
projects = {
    "Line-following robot":        np.array([1, 1, 0, 1, 0]),
    "Automatic plant watering":    np.array([1, 0, 1, 0, 0]),
    "Waste-sorting robot (color)": np.array([2, 2, 1, 0, 1]),
    "Gesture-controlled car":      np.array([3, 2, 0, 1, 1]),
    "Obstacle-avoiding robot":     np.array([1, 2, 0, 1, 0]),
}

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def recommend(team_profile, k=3):
    distances = [(dist(team_profile, features), name) for name, features in projects.items()]
    distances.sort()
    return [name for d, name in distances[:k]]

team = np.array([1, 1, 1, 0, 0])   # beginner team interested in the environment
print(recommend(team))
```

## Data sources and AI methods

| Feature | Data | AI method |
| ----------- | ----------- | ----------- |
| Project recommendation | Team questionnaires + a catalogue of projects written by teachers | Nearest neighbors (k-NN), later collaborative filtering based on which projects teams enjoyed |
| Debugging assistant | Common problems and solutions collected by teachers during club sessions | Text classification (bag of words / TF-IDF, Naive Bayes) to recognise the type of problem; a language model for guiding questions |
| Robot vision / sound | Images and sounds collected by the students themselves | Small neural networks (transfer learning), similar to [Teachable Machine](https://teachablemachine.withgoogle.com/) |
| Teacher dashboard | Logs of the sessions (projects chosen, hints asked, time spent) | Simple statistics, logistic regression to detect teams at risk of giving up |

The students collect most of the data themselves, which is also a lesson: they discover that the quality and diversity of the data decide how well the AI works.

## Challenges

* **Privacy**: students are minors. No photos of faces, no personal data leave the school, and parents are informed (in line with Moroccan law 09-08 on personal data protection).
* **Over-reliance**: the assistant must guide, not do the work for the students. Hints are given step by step.
* **Bias in data**: a model trained on few images (same lighting, same classroom) will fail elsewhere. This is discussed with the students as part of the learning.
* **Infrastructure**: unstable internet and few computers. The tools must work offline on low-cost hardware.
* **Teacher training**: teachers need time and training to use the tool. It does not replace the teacher.
* **Language**: models work less well in Darija and Amazigh than in French or English.

## What next?

* Test a first prototype in my school's robotics club and measure motivation and learning (part of my master's research).
* Build a shared catalogue of projects with other technology teachers in Morocco.
* Add support for Amazigh and voice interaction.
* Organise inter-school competitions where robots use AI models trained by the students.
* Skills and help needed: a developer for the mobile/web app, partners for low-cost kits (Arduino, micro:bit), and support from the regional academy (AREF) for training teachers.

## Acknowledgments

* Building AI course by Reaktor Innovations and the University of Helsinki
* [Teachable Machine](https://teachablemachine.withgoogle.com/) by Google, for the idea of letting students train their own models
* My students in the school robotics club, who inspired this project
* Faculty of Educational Sciences, Mohammed V University (Rabat)
