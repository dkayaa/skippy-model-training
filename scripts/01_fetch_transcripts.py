from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
import json

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

"""
JRE 
KTY 
Call Her Daddy 
YMH
Theo Von
"""

def get_label(text): 
    keyphrases = [
        '.com',
        'promo code',
        'sponsoring the podcast',
        'at checkout',
        'brought to you by',
        'thank the sponsor'
    ]

    companies = [
        'Blue Chew',
        'AG1 ',
        'White Claw', 
        'Liquid Death',
        'Stake',
        'Saily',
        'NordVPN', 
        'BetterHelp', 
        'Patreon'
    ]

    for target in keyphrases + companies:
        if target in text: 
            return 1
        
    return 0

video_ids = [
    '3CHI5u9QZQA', #Demetrious Johnson Teaches KSI TOUGH Lesson In BJJ & MMA! | MIGHTY JOURNEY!
    'g6aGhWXXJec', #KT #722- BRIAN SIMPSON + YANNIS PAPPAS
    'Htt8fFYgq3Q', #KILL TONY #579 - HANS KIM + WILLIAM MONTGOMERY + DAVID LUCAS
    '9cFCswi6Dmw', #KILL TONY #505 - TIM DILLON
    'mYvGKBCM3Ps', #Joe Rogan Experience #2332 - Oz Pearlman
    'WHdyo6IaW-w', #Paris Hilton: Sex Symbol, Slut Shaming & Sliving (Full Episode) | Call Her Daddy 
    'TamkxdBiNyY', #I’m Not Ready For A Baby… (Full Episode) | Call Her Daddy
    'SlaEaN2sm14', #This is Mia Khalifa. (Full Interview) | Call Her Daddy
    'oEfHBxEJZqU', #ICE, Iran, & The American Summer | The Tim Dillon Show #448
    '0BoTkUei9xI', #Celebrity Home Invasions & Costco | The Tim Dillon Show #395
    'AFYHXluZ6Ls', #The Return Of Shame with Dan Soder | The Tim Dillon Show #363
    'Vb3UrhyTWuo', #"Dancers" Would Drop Me Off At School w/ That Mexican OT | 2 Bears, 1 Cave
    '4Gi917jYvdg', #Steph Tolev is NAUSEATED By Cool Guys | Your Mom's House Ep. 814
    'R0vm8YlD1oo', #The Plastic Surgeon Who's Fighting The Insurance Industry | Dr. Mike
    'qb2frC7CVuA', #Do You REALLY Need A Full Body MRI? | Prenuvo CEO | Dr. Mike
    '3Q7qRw9j_lQ', #BS on Health Podcasts, Calories In/Out, & The Carnivore Diet | Layne Norton | Dr. Mike
    '2VDV87iHfKw', #David Spade Talks SNL, Chris Farley & Norm Macdonald (FULL EPISODE) | Conan O'Brien Needs A Friend
    '7r99Q0ruwOM', #Bonus Episode: THE REVENGE | Ben Shapiro 
    '677g2SsetMs', #Jeffrey Sachs: The Dark Forces Pushing Trump Into War With Iran, & Ukraine/Russia New Escalation | Tucker Carlson
    'hTSaweR8qMI', #$1 vs $500,000 Romantic Date | Mr. Beast
    'qr1AvisQcV8', #I Spent $5,000,000 So You Can Go To Space For FREE | Mark Rober
    'GbJsQk4rbBw', #Vice President JD Vance | This Past Weekend w/ Theo Von #588
    'sI2fJlZtamc', #Israel Adesanya Reacts To Kamaru Usman's Emotional Win Over Joaquin Buckley
    'nDwsX6FEVF4', #Kamaru Usman, Shut Up! | Chael Sonnen
    'pgeTa1PV_40', #Presenting my Billion Dollar Plan... | PewDiePie 
    'AU0xcQljnqA', #Can pro climber MAX out the gym? ft. Eddie Hall | Magnus Mitbo
    'qKFwte04ASI', #THEO VON: Sundae Conversation with Caleb Pressley | Sundae Conversation
    'VBsga-k3XuM', #Exclusive: Sean O'Malley Opens Up About Merab Loss, Fires Back at Garbrandt, Teases Quick Turnaround | Ariel Helwani Show
    'smuVx0vaJT4', #"I'm a Bad Matchup & He KNOWS It!" Tom Aspinall On WHY Jon Jones Is Ducking Him | Mighty
    '0d6Vkc93KeA', #SMNTY Interviews: Raquel Willis | STUFF MOM NEVER TOLD YOU
    'eosMB1jfa_0', #EP24 Manifesting the Muse with Rick | Dan Carlin 
    'FHg_Ko6hXQM', #Comedian Liz Miele's Rollercoaster Scare | Kevin Nealon
    'ypz_Wz0n1Lw', #I Tried To Break The Record Up The World's Longest Stairs | Goran Winblad
    'yIEepbaGeY8', #Why It’s Too Early To Give Up On Lewis Hamilton. | Tommo 
    'lQGfSLhELjM', #The Lando Norris Question: Why McIlroy & Federer Have The Answers. | Tommo
    '3CHI5u9QZQA', #Demetrious Johnson Teaches KSI TOUGH Lesson In BJJ & MMA! | MIGHTY JOURNEY!
    '3M3v1stjuss', #Doctor Reacts To Health Advice From TikTok
    '3ZTGwcHQfLY', #I Tried To Make Something In America (The Smarter Scrubber Experiment) - Smarter Every Day 308
    '7arjH-sGWFM', #Making the world's most powerful Red Bull | NileBlue
    '5DJA3YS7FWg', #Trial Lawyer Reacts to Diddy Trial Circus | LegalEagle
    'Xlzq_EQHKMY', #LUKA DONCIC: Sundae Conversation with Caleb Pressley | Sundae Conversation
    'j9Qm6_lEdcQ', #Dakota Johnson Is Not Okay While Eating Spicy Wings | Hot Ones
    'j-ltAfx_dvI', #BELLA HADID | CHICKEN SHOP DATE
    '-WNnWVH_EoY', #ERIC ANDRE | CHICKEN SHOP DATE
    'vCz3VCQ40dM', #TRY NOT TO LAUGH WITH WILL SMITH | KSI 
    'aB9Nh8OISdk', #Running 100 Miles in 19 hours and 13 minutes | Rocky Racoon
    'cX-4nRTanBM', #The BEST Daily in 2025? New Balance Rebel v5 Honest Review
    'mb_W1SG9UWo', #My LAST WORKOUTS Before Racing An ALL OUT 5K!
    'h43lgA0UcvM', #Stop Chasing The Next 'Best' Thing And Start Seeing Results | John Lancaster
    'Bpfv9OlpB9U', #Tony Hinchcliffe Gets Humbled By David Spade
    '6lVuOX4surM', #100 Men v 1 Gorilla | Ep 274 | Bad Friends
    'cCi_jX0lj44', #Tito Cheeto is Back! | Ep 160 ft. Ryan Sickler | Bad Friends
    'lN3bMDc95YM', #Alfred Packer Ate Five Guys, No Burgers | Ep 278
    'OYOAbCQ6DUo', #I Cheated A Marathon Using An Exoskeleton | Chris Howett
    'wFy0hV3O3rE', #The Hurt we Choose | The rise of ULTRA RUNNING
    'PvD_vwImGNs', #Why Rome Collapsed - Barry Strauss | Trigonometry
    'UBCvspUhUPo', #The Israel-Iran War Could Get Much Worse. Here’s How.
    'TNIgFE-JS8Q', #Saudi Leader Makes Host Go Quiet with This Chilling Warning about Iran’s Leader | Rubin Report 
    '_Ngbr2ibxyM', #Highway Bike Racing RAW and UNCUT // Bike for Brain Health 2025
    'fRBiPwA_EfM', #Zwift vs Rouvy in 2025: Which Is Best for Indoor Cycling?
    '1E3tv_3D95g', #WWDC 2025 Impressions: Liquid Glass! | MKBHD
    'p9AGycqfvVQ', #Ep 563 - No Nut Clarity (feat. Chris O'Connor & Dave Temple) | Matt and Shane Secret Podcast
    'PbWVbQQwWJo', #Bernie Sanders Rips DC Corruption, The Israel Lobby, & Reveals How Billionaires Buy Politicians | Flagrant
    'espTOMuQz54', #Daniel Cormier SHOCKED Islam Makhachev only has 3 MORE FIGHTS in UFC including Jack Della Maddalena
    'tFM6L6TopsM', #Tucker and Steve Bannon Respond to Israel’s War on Iran and How It Could Destroy MAGA Forever
    'KBweCEFejLs', #Dave Smith | Trump and Iran w/ Scott Horton | Part Of The Problem 1276
    'F5vvy2S90KQ', #The Failed Assassination Of Donald Trump w/ Scott Horton | Part Of The Problem 1143
    '8ijMBtAe8Tc', #Drugged In Colombia, Escaping Jail & Defeating UFC Wrestling - Craig Jones
    'ny_S1u_PKkk', #Why Nice Guys Usually End Up Being Lonely & Bitter - Dr Robert Glover
    'RSfMFwHKoZs', #We Should Not Die as Traders · Patrick Petersson | Chat with Traders
    '9PldqVePztM', #This $269 Gaming PC can play any game! | Linus Tech Tips
    'JxrlCBkCLWA', #Cooling a PC with a Fire Truck… Literally | Linus Tech Tips
    'KLkwMMu9qlk', #Reacting to INSANE Chinese Gaming Setups | Linus Tech Tips
    'll-n_wUIzeA', #How to Build a Gaming PC in 2025 | Austin Evans
    'N4cdKkjbdtE', #This 2024 Tech is AWESOME | Austin Evans
    'ZAWaFMFJu1U', #No Bull w/ Johnny Pemberton | Your Mom's House Ep. 763
    '7IALs32-5Sc', #Saturday Night Live Secrets & Norm Macdonald w/ Kevin Nealon | 2 Bears, 1 Cave
    'vsp69jYlYsg', #Your Mom's House Podcast w/ Andrew Tate - Ep.636
    'ROJ3PdDmirY', #How'd They Make This Mistake? | Low Level
    'j2MBF34-cUg', #The World's Strongest Man Kicks My Leg at 100% 😳 | Tom Aspinall
    'HoRY-o2sRdA', #Israel, Hamas, and the Battle for Civilization | Sam Harris & Douglas Murray
    'CfcxpDgTtMQ', #Boys Time, Alex Cooper, Herman Herman | Fact Check from Alex Cooper
    'ftln3QblqFk', #Armchair Anonymous: Church | Armchair Expert with Dax Shepard
    'rZkMpVLcVsg', #How Relationships Shape Your Brain | Dr. Allan Schore | Huberman Lab
    'qNzl12g0Dd8', #Efforts & Challenges in Promoting Public Health | Dr. Vivek Murthy | Huberman Lab
    'WGrGFDGYWm8', #It’s SPICY GIRL SUMMER (how to do it right) | Ep. 415 | Girls Gotta Eat
    '4lZXV5FzVhU', #Men vs. Women in Breakups feat. Ricky Liorti | Ep. 390 | Girls Gotta Eat
    'qdqMe5jnH-4', #Country Clubs and Sex Toys with Heather McMahan (and Jeff) | Ep. 320 | Girls Gotta Eat
    'VzrsG9Alhrc', #My Favorite Murder 483 - Those Pants, That Hand
    't3snQ8sJKL8', #Negra on R.I.C.O. Charges, Her History with Baldacci, Eyekon vs Foo Community & More | No Jumper
    'A8mI4nd2_2k', #Go Yayo Goes Crazy on Sauce Walka, Says F*** MO3, Being Blackballed & More | No Jumper
    'Gp6hjL2aAPg', #GASLIT BY A GAS LEAK Ft. Good Children | Drew Afualo | The Comment Section Ep. 119
    'VbMTV6AGA9I', #(Fe)Male Gazing | Drew Afualo ft. Brittany Broski | THE COMMENT SECTION EP 5
    '8ftGIMSRf5k', #From $500 to a Lash Empire: Power Moves & Purpose ft. Yris Palmer | Khloé In Wonder Land Ep. 19
    'ZQ7ThhoWoOE', #How to Actually Find Love in 2025 ft. Logan Ury | Khloé In Wonder Land Ep. 10
    'W6gpX-yJnIc', #Coulda Been Records PHILADELPHIA Auditions hosted by Druski | Druski 
    'bklmi6Zf4LQ', #Coulda Been Records HOUSTON Auditions pt. 2 hosted by Druski | Druski 
    'lE5qrGT7ZZg', #Cynthia Erivo: "I Was Working To Prove That I Was Worth Loving" #1 Way To Know it's time to LEAVE! | Jay Shetty On Purpose 
    'GuRfdtyEtSo', #Jamie Kern Lima's Billion-Dollar Journey: From Denny's Waitress to L'Oréal's First Female CEO | Jay Shetty On Purpose 
    'BwjnG45zO5U', #Tony Robbins ON: How To BRAINWASH Yourself For Success & Destroy NEGATIVE THOUGHTS! | Jay Shetty On Purpose
]

ytt_api =YouTubeTranscriptApi()

for video_id in video_ids:
    try:
        fetched_transcript = ytt_api.fetch(video_id)
    except Exception as e: 
        print(e)

    window_size = 20
    stride =10
    segments = []
    for i in range(0, len(fetched_transcript) - window_size, stride):
        segment_text = " ".join([snippet.text for snippet in fetched_transcript[i:i+window_size]])
        segment_start = fetched_transcript[i].start 
        segments.append({
            'text':segment_text,
            'start':segment_start, 
            'label': get_label(segment_text) #default label
        })

    #merged_transcript = ' '.join([snippet.text for snippet in fetched_transcript])

    with open(TRANSCRIPTS_DIR / f"{video_id}.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

