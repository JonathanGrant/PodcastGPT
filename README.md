# PodcastGPT

## Text-to-Speech Voices

The project supports multiple TTS providers. Google TTS now defaults to
Chirp 3 HD voices for male and female synthesis. The available Chirp 3 HD
voices include:

* Achernar
* Achird
* Algenib
* Algieba
* Alnilam
* Aoede
* Autonoe
* Callirrhoe
* Charon
* Despina
* Enceladus
* Erinome
* Fenrir
* Gacrux
* Iapetus
* Kore
* Laomedeia
* Leda
* Orus
* Puck
* Pulcherrima
* Rasalgethi
* Sadachbia
* Sadaltager
* Schedar
* Sulafat
* Umbriel
* Vindemiatrix
* Zephyr
* Zubenelgenubi

`GoogleTTS` will use **Leda** for `WOMAN` and **Charon** for `MAN` by default.

The MoneyStuff runner randomly chooses a different Chirp 3 voice for each
paragraph of an article.

### Overcast Uploads

If you have an Overcast Plus account, MoneyStuff can attempt to upload the
generated MP3 directly to your Overcast uploads page. Set the environment
variables `OVERCAST_EMAIL` and `OVERCAST_PASSWORD` before running the script.
The upload process scrapes the web form and may fail if Overcast changes its
interface.
